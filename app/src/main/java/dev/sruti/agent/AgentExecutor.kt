package dev.sruti.agent

import dev.sruti.llm.ChatEvent
import dev.sruti.llm.ChatMessage
import dev.sruti.llm.ChatParams
import dev.sruti.llm.ChatSession
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive

/** Everything the agent does is reported, so the trace is auditable. */
sealed interface AgentEvent {
    data class StepStarted(val step: Int, val maxSteps: Int) : AgentEvent
    data class CandidatesRetrieved(val tools: List<Tool>) : AgentEvent
    data class ToolChosen(val tool: Tool) : AgentEvent
    data class ArgumentsExtracted(val tool: Tool, val arguments: Map<String, String>) : AgentEvent
    data class ConfirmationRequired(val tool: Tool, val arguments: Map<String, String>) : AgentEvent
    data class ToolFinished(
        val tool: Tool,
        val arguments: Map<String, String>,
        val result: ToolResult,
    ) : AgentEvent

    @JvmInline
    value class AnswerToken(val piece: String) : AgentEvent

    data class Finished(val steps: Int, val reason: Reason) : AgentEvent {
        enum class Reason { Answered, StepLimit, Declined, NoToolApplies, Failed }
    }

    @JvmInline
    value class Failed(val message: String) : AgentEvent
}

/** Asked before any tool that mutates state runs. Return false to decline. */
typealias ConfirmationRequest = suspend (Tool, Map<String, String>) -> Boolean

/**
 * Runs a request as a sequence of narrow, constrained model calls.
 *
 * The model is never asked to plan. It is asked one question at a time — "which
 * of these few tools", then "fill these slots" — and the loop between those
 * questions is ordinary code with a hard step ceiling. That inversion is the whole
 * design: a 1–2B model cannot hold a multi-step plan, but it can answer a narrow
 * question, and a state machine can be the thing that remembers.
 *
 * Grammars make each answer structurally valid by construction. What they cannot
 * do is make it *correct* — a small model will confidently pick the wrong tool —
 * so candidates are narrowed by retrieval first, arguments are validated against
 * the schema afterwards, and anything that mutates state is confirmed by the user.
 */
class AgentExecutor(
    private val session: ChatSession,
    private val registry: ToolRegistry,
    private val maxSteps: Int = 5,
) {

    fun run(
        request: String,
        allowedTiers: Set<ToolTier> = setOf(ToolTier.InApp),
        confirm: ConfirmationRequest = { _, _ -> true },
        params: ChatParams = ChatParams(temperature = 0.2f, maxTokens = 256),
    ): Flow<AgentEvent> = flow {
        val observations = mutableListOf<String>()

        for (step in 1..maxSteps) {
            emit(AgentEvent.StepStarted(step, maxSteps))

            // Accuracy collapses as the candidate list grows at this model size,
            // so the model never sees the whole registry.
            val candidates = registry.retrieve(request, allowedTiers)
            emit(AgentEvent.CandidatesRetrieved(candidates))

            if (candidates.isEmpty()) {
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.NoToolApplies))
                return@flow
            }

            val choice = chooseTool(request, candidates, observations, params)
            if (choice == null || choice == GbnfGrammar.NO_TOOL) {
                // The model declining is a legitimate outcome, not a failure: it
                // is why the choice grammar always includes an escape hatch.
                answer(request, observations, params)
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Answered))
                return@flow
            }

            val tool = registry.find(choice)
            if (tool == null) {
                emit(AgentEvent.Failed("model chose an unknown tool: $choice"))
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Failed))
                return@flow
            }
            emit(AgentEvent.ToolChosen(tool))

            val extracted = extractArguments(request, tool, observations, params)
            if (extracted == null) {
                emit(AgentEvent.Failed("could not read arguments for ${tool.name}"))
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Failed))
                return@flow
            }

            // Deterministic validation rather than a second model pass. The
            // grammar already guarantees shape, and a model that picked the wrong
            // tool is not the thing to ask whether it picked the wrong tool.
            val invalid = validateToolCall(tool, extracted)
            if (invalid != null) {
                emit(AgentEvent.Failed(invalid))
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Failed))
                return@flow
            }
            emit(AgentEvent.ArgumentsExtracted(tool, extracted))

            if (tool.requiresConfirmation) {
                emit(AgentEvent.ConfirmationRequired(tool, extracted))
                if (!confirm(tool, extracted)) {
                    emit(AgentEvent.ToolFinished(tool, extracted, ToolResult.Declined))
                    emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Declined))
                    return@flow
                }
            }

            val result = runCatching { tool.execute(extracted) }
                .getOrElse { ToolResult.Failure(it.message ?: "tool threw") }
            emit(AgentEvent.ToolFinished(tool, extracted, result))

            observations += when (result) {
                is ToolResult.Success -> "${tool.name} returned: ${result.output.take(OBSERVATION_LIMIT)}"
                is ToolResult.Failure -> "${tool.name} failed: ${result.message}"
                ToolResult.Declined -> "${tool.name} was declined by the user"
            }

            if (result is ToolResult.Failure) {
                // Retrying a failed call with the same model and the same request
                // reliably produces the same failure; better to report it.
                answer(request, observations, params)
                emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Failed))
                return@flow
            }

            // One successful tool call is the design point. Chaining is where a
            // model this size loses the thread, so the loop is capped and each
            // pass re-derives from the observations rather than from a held plan.
            answer(request, observations, params)
            emit(AgentEvent.Finished(step, AgentEvent.Finished.Reason.Answered))
            return@flow
        }

        emit(AgentEvent.Finished(maxSteps, AgentEvent.Finished.Reason.StepLimit))
    }

    // --- narrow questions ----------------------------------------------------

    private suspend fun chooseTool(
        request: String,
        candidates: List<Tool>,
        observations: List<String>,
        params: ChatParams,
    ): String? {
        val prompt = buildString {
            appendLine("Available tools:")
            candidates.forEach { appendLine("- ${it.signature()}") }
            appendLine("- ${GbnfGrammar.NO_TOOL} — no tool is needed")
            appendLine()
            if (observations.isNotEmpty()) {
                appendLine("Already done:")
                observations.forEach { appendLine("- $it") }
                appendLine()
            }
            appendLine("Request: $request")
            append("Reply with exactly one tool name.")
        }

        val raw = collect(prompt, params.copy(maxTokens = 16), GbnfGrammar.toolChoice(candidates))
        return raw.trim().trim('"').ifBlank { null }
    }

    private suspend fun extractArguments(
        request: String,
        tool: Tool,
        observations: List<String>,
        params: ChatParams,
    ): Map<String, String>? {
        val prompt = buildString {
            appendLine("Tool: ${tool.signature()}")
            tool.parameters.forEach { p ->
                val suffix = if (p.values.isNotEmpty()) " (one of: ${p.values.joinToString(", ")})" else ""
                appendLine("- ${p.name}: ${p.description}$suffix")
            }
            appendLine()
            if (observations.isNotEmpty()) {
                appendLine("Already done:")
                observations.forEach { appendLine("- $it") }
                appendLine()
            }
            appendLine("Request: $request")
            append("Output the arguments as JSON.")
        }

        val raw = collect(prompt, params.copy(maxTokens = 128), GbnfGrammar.toolCall(tool))
        return parseToolArguments(raw)
    }

    /**
     * Streams a final, unconstrained answer grounded in what actually happened.
     *
     * Free text on purpose: the tool results are already facts, and forcing the
     * summary through a grammar would gain nothing while making it stilted.
     */
    private suspend fun FlowCollector<AgentEvent>.answer(
        request: String,
        observations: List<String>,
        params: ChatParams,
    ) {
        val prompt = buildString {
            if (observations.isEmpty()) {
                appendLine("Answer this request directly and briefly.")
            } else {
                appendLine("These actions were performed:")
                observations.forEach { appendLine("- $it") }
                appendLine()
                appendLine("Answer the request using only the results above.")
            }
            appendLine()
            append("Request: $request")
        }

        session.reset()
        session.generate(
            messages = listOf(ChatMessage(ChatMessage.Role.User, prompt)),
            params = params,
        ).collect { event ->
            if (event is ChatEvent.Token) {
                emit(AgentEvent.AnswerToken(event.piece))
            }
        }
    }

    /** Collects a constrained completion as a single string. */
    private suspend fun collect(
        prompt: String,
        params: ChatParams,
        grammar: String?,
    ): String {
        // A fresh single-turn context each time. Carrying conversational history
        // into a constrained slot-filling question measurably degrades it.
        session.reset()
        return session
            .generate(
                messages = listOf(ChatMessage(ChatMessage.Role.User, prompt)),
                params = params,
                grammar = grammar,
            )
            .toList()
            .filterIsInstance<ChatEvent.Token>()
            .joinToString("") { it.piece }
    }

    private companion object {
        /** Observations are fed back into a small context; long output crowds it out. */
        const val OBSERVATION_LIMIT = 500
    }
}

private val argumentJson = Json { ignoreUnknownKeys = true }

/**
 * Parses a grammar-constrained JSON object into plain string values.
 *
 * A top-level function rather than a method because it is pure, and because the
 * checks that stand between a model's output and something running should be
 * testable without loading a model.
 */
internal fun parseToolArguments(raw: String): Map<String, String>? {
    val text = raw.trim()
    if (text.isEmpty()) return null
    return runCatching {
        val obj = argumentJson.parseToJsonElement(text) as? JsonObject ?: return null
        obj.mapValues { (_, value) ->
            // Unquoted, so a tool receives "5" rather than "\"5\"".
            (value as? JsonPrimitive)?.content ?: value.toString()
        }
    }.getOrNull()
}

/**
 * Returns a message when a call is unusable, or null when it is fine.
 *
 * The grammar guarantees the shape of a call. Nothing guarantees the model put
 * anything sensible inside it, so this is the last check before execution.
 */
internal fun validateToolCall(tool: Tool, arguments: Map<String, String>): String? {
    tool.parameters.filter { it.required }.forEach { p ->
        if (arguments[p.name].isNullOrBlank()) {
            return "${tool.name} needs '${p.name}' but it was missing or empty"
        }
    }

    tool.parameters.forEach { p ->
        val value = arguments[p.name] ?: return@forEach
        when (p.type) {
            ToolParameter.Type.Integer ->
                if (value.toLongOrNull() == null) {
                    return "${tool.name}.${p.name} must be an integer, got '$value'"
                }
            ToolParameter.Type.Number ->
                if (value.toDoubleOrNull() == null) {
                    return "${tool.name}.${p.name} must be a number, got '$value'"
                }
            ToolParameter.Type.Boolean ->
                if (value != "true" && value != "false") {
                    return "${tool.name}.${p.name} must be true or false, got '$value'"
                }
            ToolParameter.Type.Enum ->
                if (p.values.isNotEmpty() && value !in p.values) {
                    return "${tool.name}.${p.name} must be one of ${p.values}, got '$value'"
                }
            ToolParameter.Type.String -> Unit
        }
    }

    // The grammar should make these unreachable, so their presence means something
    // upstream is wrong and executing anyway would hide it.
    val unknown = arguments.keys - tool.parameters.map { it.name }.toSet()
    if (unknown.isNotEmpty()) {
        return "${tool.name} received unexpected arguments: ${unknown.joinToString()}"
    }

    return null
}
