package dev.sruti.agent

import dev.sruti.llm.ChatEvent
import dev.sruti.llm.ChatMessage
import dev.sruti.llm.ChatParams
import dev.sruti.llm.ChatSession
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList

/**
 * Runs a templated skill.
 *
 * The whole point is what this does *not* ask the model. Which tools to use, in
 * what order, and where each result goes are all written down in the skill. The
 * model is asked exactly one question — what are the parameter values in this
 * request — under a grammar that makes the answer structurally valid. Everything
 * after that is ordinary code.
 *
 * That is why a skill is worth having: it converts a task the model would fail
 * at, because it requires holding a plan, into a task it is genuinely good at,
 * which is pulling a filename out of a sentence.
 */
class SkillRunner(
    private val session: ChatSession,
    private val registry: ToolRegistry,
) {

    fun run(
        skill: Skill,
        request: String,
        allowedTiers: Set<ToolTier> = setOf(ToolTier.InApp),
        confirm: ConfirmationRequest = { _, _ -> true },
        params: ChatParams = ChatParams(temperature = 0.2f, maxTokens = 256),
    ): Flow<AgentEvent> = flow {
        val missing = skill.missingTools(registry)
        if (missing.isNotEmpty()) {
            emit(AgentEvent.Failed("this skill needs tools that are not available: " +
                missing.joinToString(", ")))
            emit(AgentEvent.Finished(0, AgentEvent.Finished.Reason.Failed))
            return@flow
        }

        // Checked before anything runs rather than at the step that needs it: a
        // skill that stops halfway leaves the workspace in a state the user did
        // not ask for.
        val blocked = skill.steps.mapNotNull { registry.find(it.tool) }
            .filterNot { it.tier in allowedTiers }
        if (blocked.isNotEmpty()) {
            emit(AgentEvent.Failed("this skill needs the shell, which is turned off"))
            emit(AgentEvent.Finished(0, AgentEvent.Finished.Reason.Failed))
            return@flow
        }

        val slots = if (skill.parameters.isEmpty()) {
            emptyMap()
        } else {
            extractParameters(skill, request, params)
                ?: run {
                    emit(AgentEvent.Failed("could not read the values this skill needs"))
                    emit(AgentEvent.Finished(0, AgentEvent.Finished.Reason.Failed))
                    return@flow
                }
        }

        val missingRequired = skill.parameters
            .filter { it.required && slots[it.name].isNullOrBlank() }
            .map { it.name }
        if (missingRequired.isNotEmpty()) {
            emit(AgentEvent.Failed("missing: " + missingRequired.joinToString(", ")))
            emit(AgentEvent.Finished(0, AgentEvent.Finished.Reason.Failed))
            return@flow
        }

        // Grows as steps complete, so a later step can reference {{step1}}.
        val values = slots.toMutableMap()
        val observations = mutableListOf<String>()

        skill.steps.forEachIndexed { index, step ->
            val stepNumber = index + 1
            emit(AgentEvent.StepStarted(stepNumber, skill.steps.size))

            val tool = registry.find(step.tool) ?: return@flow
            emit(AgentEvent.ToolChosen(tool))

            val arguments = step.arguments.mapValues { (_, template) ->
                applySlots(template, values)
            }

            // A slot that never got a value would otherwise be passed through as
            // the literal text "{{path}}", which reads as a bizarre filename
            // rather than as the missing input it is.
            val unresolved = arguments.values.flatMap { slotsIn(it) }.distinct()
            if (unresolved.isNotEmpty()) {
                emit(AgentEvent.Failed("nothing supplied for: " + unresolved.joinToString(", ")))
                emit(AgentEvent.Finished(stepNumber, AgentEvent.Finished.Reason.Failed))
                return@flow
            }

            val invalid = validateToolCall(tool, arguments)
            if (invalid != null) {
                emit(AgentEvent.Failed(invalid))
                emit(AgentEvent.Finished(stepNumber, AgentEvent.Finished.Reason.Failed))
                return@flow
            }
            emit(AgentEvent.ArgumentsExtracted(tool, arguments))

            if (tool.requiresConfirmation) {
                emit(AgentEvent.ConfirmationRequired(tool, arguments))
                if (!confirm(tool, arguments)) {
                    emit(AgentEvent.ToolFinished(tool, arguments, ToolResult.Declined))
                    emit(AgentEvent.Finished(stepNumber, AgentEvent.Finished.Reason.Declined))
                    return@flow
                }
            }

            val result = runCatching { tool.execute(arguments) }
                .getOrElse { ToolResult.Failure(it.message ?: "tool threw") }
            emit(AgentEvent.ToolFinished(tool, arguments, result))

            when (result) {
                is ToolResult.Success -> {
                    values["step$stepNumber"] = result.output
                    observations += "${tool.name} returned: ${result.output.take(OBSERVATION_LIMIT)}"
                }
                is ToolResult.Failure -> {
                    // A skill is a fixed sequence, so a failed step means the rest
                    // would run on data that does not exist.
                    observations += "${tool.name} failed: ${result.message}"
                    answer(skill, request, observations, params)
                    emit(AgentEvent.Finished(stepNumber, AgentEvent.Finished.Reason.Failed))
                    return@flow
                }
                ToolResult.Declined -> {
                    emit(AgentEvent.Finished(stepNumber, AgentEvent.Finished.Reason.Declined))
                    return@flow
                }
            }
        }

        answer(skill, request, observations, params)
        emit(AgentEvent.Finished(skill.steps.size, AgentEvent.Finished.Reason.Answered))
    }

    /**
     * The one question the model is asked.
     *
     * Constrained to a JSON object with exactly the skill's parameter names, so
     * the reply cannot be shaped wrongly — only wrong, which the caller then
     * checks.
     */
    private suspend fun extractParameters(
        skill: Skill,
        request: String,
        params: ChatParams,
    ): Map<String, String>? {
        val prompt = buildString {
            appendLine("Task: ${skill.description}")
            appendLine()
            appendLine("Read these values out of the request:")
            skill.parameters.forEach { p ->
                appendLine("- ${p.name}: ${p.description}")
            }
            appendLine()
            appendLine("Request: $request")
            append("Output the values as JSON.")
        }

        val grammar = GbnfGrammar.slots(skill.parameters)

        session.reset()
        val raw = session
            .generate(
                messages = listOf(ChatMessage(ChatMessage.Role.User, prompt)),
                params = params.copy(maxTokens = 128),
                grammar = grammar,
            )
            .toList()
            .filterIsInstance<ChatEvent.Token>()
            .joinToString("") { it.piece }

        return parseToolArguments(raw)
    }

    private suspend fun FlowCollector<AgentEvent>.answer(
        skill: Skill,
        request: String,
        observations: List<String>,
        params: ChatParams,
    ) {
        val prompt = buildString {
            if (skill.body.isNotBlank()) {
                appendLine(skill.body)
                appendLine()
            }
            appendLine("These actions were performed:")
            observations.forEach { appendLine("- $it") }
            appendLine()
            appendLine("Answer the request using only the results above.")
            appendLine()
            append("Request: $request")
        }

        session.reset()
        session.generate(
            messages = listOf(ChatMessage(ChatMessage.Role.User, prompt)),
            params = params,
        ).collect { event ->
            if (event is ChatEvent.Token) emit(AgentEvent.AnswerToken(event.piece))
        }
    }

    private companion object {
        const val OBSERVATION_LIMIT = 600
    }
}
