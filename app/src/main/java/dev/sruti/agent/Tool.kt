package dev.sruti.agent

/**
 * What a tool needs from the user before it may run.
 *
 * Tiers are about capability, not danger: a tool that shells out can do anything
 * the shell can, so it is gated on an explicit, informed opt-in rather than on a
 * judgement about any particular command.
 */
enum class ToolTier {
    /** App sandbox, picked documents, network fetch. No special permission. */
    InApp,

    /** A real POSIX shell via Termux. Off until the user turns it on. */
    Shell,
}

/** A single tool parameter. */
data class ToolParameter(
    val name: String,
    val type: Type,
    val description: String,
    val required: Boolean = true,
    /** For [Type.Enum]; ignored otherwise. */
    val values: List<String> = emptyList(),
) {
    enum class Type { String, Integer, Number, Boolean, Enum }
}

/** The outcome of running a tool. */
sealed interface ToolResult {
    /** [output] is fed back to the model, so it must be short and factual. */
    data class Success(val output: String) : ToolResult

    data class Failure(val message: String) : ToolResult

    /** The user declined at the confirmation step. */
    data object Declined : ToolResult
}

/**
 * A capability the agent can invoke.
 *
 * Descriptions are written for a 1–2B model, which means concrete and short. A
 * model this size does not infer intent from prose; it matches surface patterns.
 */
data class Tool(
    val name: String,
    val description: String,
    val parameters: List<ToolParameter> = emptyList(),
    val tier: ToolTier = ToolTier.InApp,
    /**
     * Whether the user must approve each call.
     *
     * True for anything that writes, sends or executes. Reading is not gated —
     * requiring approval for every read would train the user to tap through
     * confirmations without reading them, which is worse than not asking.
     */
    val requiresConfirmation: Boolean = false,
    /** Words that should retrieve this tool, beyond those in its name. */
    val keywords: List<String> = emptyList(),
    val execute: suspend (Map<String, String>) -> ToolResult,
) {
    /** One-line form used in prompts. Kept terse; context is scarce. */
    fun signature(): String {
        val params = parameters.joinToString(", ") { p ->
            if (p.required) p.name else "${p.name}?"
        }
        return "$name($params) — $description"
    }
}

/**
 * The set of tools available to the agent.
 *
 * Retrieval matters as much as the tools themselves. Accuracy at this model size
 * collapses as the candidate list grows, so the executor never shows the model
 * everything — it shows a handful of plausible options and asks one question.
 */
class ToolRegistry(tools: List<Tool> = emptyList()) {

    private val byName = tools.associateBy { it.name }

    val all: List<Tool> get() = byName.values.toList()

    fun find(name: String): Tool? = byName[name]

    fun enabledFor(allowedTiers: Set<ToolTier>): List<Tool> =
        all.filter { it.tier in allowedTiers }

    /**
     * Ranks tools against a request and returns the best few.
     *
     * Lexical overlap rather than embeddings: it needs no second model, no extra
     * memory, and is predictable enough to debug. A wrong candidate set is
     * recoverable — the model simply reports it cannot help — whereas a silently
     * mis-ranked embedding is not.
     */
    fun retrieve(
        request: String,
        allowedTiers: Set<ToolTier>,
        limit: Int = 5,
    ): List<Tool> {
        val terms = tokenize(request)
        if (terms.isEmpty()) return enabledFor(allowedTiers).take(limit)

        return enabledFor(allowedTiers)
            .map { tool -> tool to score(tool, terms) }
            .sortedWith(compareByDescending<Pair<Tool, Int>> { it.second }.thenBy { it.first.name })
            .take(limit)
            .map { it.first }
    }

    private fun score(tool: Tool, terms: Set<String>): Int {
        val nameTerms = tokenize(tool.name)
        val keywordTerms = tool.keywords.flatMap { tokenize(it) }.toSet()
        val descriptionTerms = tokenize(tool.description)

        var score = 0
        terms.forEach { term ->
            // A name match is the strongest signal, then curated keywords, then
            // incidental description overlap.
            if (term in nameTerms) score += 5
            if (term in keywordTerms) score += 3
            if (term in descriptionTerms) score += 1
        }
        return score
    }

    private fun tokenize(text: String): Set<String> =
        text.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length > 2 && it !in STOP_WORDS }
            .toSet()

    private companion object {
        val STOP_WORDS = setOf(
            "the", "and", "for", "with", "that", "this", "from", "into", "over",
            "any", "all", "you", "your", "can", "get", "use", "using",
        )
    }
}
