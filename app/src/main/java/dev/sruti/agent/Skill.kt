package dev.sruti.agent

/**
 * A named recipe for a task the agent can carry out.
 *
 * This is the plan's answer to the central problem: a 1–2B model cannot hold a
 * multi-step plan, so the plan is written down instead. A skill supplies the
 * sequence; the model only fills the blanks in it. That inverts where the
 * intelligence has to live — from the model, which does not have enough, to the
 * author, who does.
 *
 * A skill with [steps] runs deterministically: the model is asked exactly one
 * question — fill these slots — and the ordering, the tools and the data flow
 * between them are all fixed. A skill without steps is guidance only; its body
 * is prepended to the ordinary agent loop, which is weaker but takes a sentence
 * to write.
 */
data class Skill(
    val name: String,
    val description: String,
    val keywords: List<String> = emptyList(),
    val parameters: List<SkillParameter> = emptyList(),
    val steps: List<SkillStep> = emptyList(),
    /** Free-form instructions; the whole skill when [steps] is empty. */
    val body: String = "",
    /** True for skills that shipped with the app and cannot be edited. */
    val builtIn: Boolean = false,
) {
    val isTemplated: Boolean get() = steps.isNotEmpty()

    /** Highest tier any step needs, so the shell requirement is visible up front. */
    fun requiredTiers(registry: ToolRegistry): Set<ToolTier> =
        steps.mapNotNull { registry.find(it.tool)?.tier }.toSet()

    /** Tools this skill names that the registry does not have. */
    fun missingTools(registry: ToolRegistry): List<String> =
        steps.map { it.tool }.distinct().filter { registry.find(it) == null }
}

data class SkillParameter(
    val name: String,
    val description: String,
    val required: Boolean = true,
)

/**
 * One tool invocation in a skill.
 *
 * [arguments] values may contain `{{slot}}` references — either a skill
 * parameter, or `{{stepN}}` for the output of an earlier step. Substitution is
 * plain text: the model never sees the template, only the question of what the
 * parameters are.
 */
data class SkillStep(
    val tool: String,
    val arguments: Map<String, String>,
)

/** A skill file could not be read. */
class SkillFormatException(message: String) : IllegalArgumentException(message)

/**
 * Reads the skill file format.
 *
 * Markdown with a YAML-ish frontmatter block, because a skill has to be
 * writable by hand in a text field on a phone and shareable as a single file.
 * The parser is deliberately small and handles a strict subset rather than
 * pulling in a YAML library to accept things the format does not need.
 *
 * ```
 * ---
 * name: summarise-notes
 * description: Read a file and write a summary beside it
 * keywords: summarise, digest
 * parameters:
 *   - path: File to summarise
 * steps:
 *   - read_file(path={{path}})
 *   - write_file(path={{path}}.summary.md, content={{step1}})
 * ---
 *
 * Any prose here is passed to the model as extra guidance.
 * ```
 */
object SkillParser {

    private val STEP = Regex("""^([A-Za-z0-9_]+)\s*\((.*)\)$""")

    fun parse(text: String, builtIn: Boolean = false): Skill {
        val normalized = text.replace("\r\n", "\n")
        val lines = normalized.split('\n')

        val open = lines.indexOfFirst { it.trim() == "---" }
        if (open != 0) {
            throw SkillFormatException("a skill must start with a --- frontmatter block")
        }
        val close = lines.drop(1).indexOfFirst { it.trim() == "---" }
        if (close < 0) throw SkillFormatException("frontmatter block is not closed with ---")

        val front = lines.subList(1, close + 1)
        val body = lines.drop(close + 2).joinToString("\n").trim()

        var name = ""
        var description = ""
        var keywords = emptyList<String>()
        val parameters = mutableListOf<SkillParameter>()
        val steps = mutableListOf<SkillStep>()

        // Two-state walk rather than a general parser: a key introduces a value
        // or a list, and list items belong to the most recent key.
        var section = ""
        front.forEach { raw ->
            val line = raw.trimEnd()
            if (line.isBlank()) return@forEach

            val item = line.trimStart()
            if (item.startsWith("- ")) {
                val content = item.removePrefix("- ").trim()
                when (section) {
                    "parameters" -> parameters += parseParameter(content)
                    "steps" -> steps += parseStep(content)
                    else -> Unit
                }
                return@forEach
            }

            val colon = line.indexOf(':')
            if (colon <= 0) return@forEach
            val key = line.take(colon).trim().lowercase()
            val value = line.substring(colon + 1).trim()
            section = key

            when (key) {
                "name" -> name = value
                "description" -> description = value
                "keywords" -> keywords = value.split(',', ' ')
                    .map { it.trim().trim('[', ']', '"') }
                    .filter { it.isNotBlank() }
                else -> Unit
            }
        }

        if (name.isBlank()) throw SkillFormatException("a skill needs a name")
        if (description.isBlank()) throw SkillFormatException("a skill needs a description")

        return Skill(
            name = name,
            description = description,
            keywords = keywords,
            parameters = parameters,
            steps = steps,
            body = body,
            builtIn = builtIn,
        )
    }

    /** `- path: File to summarise`, with a trailing `?` marking it optional. */
    private fun parseParameter(content: String): SkillParameter {
        val colon = content.indexOf(':')
        val rawName = (if (colon > 0) content.take(colon) else content).trim()
        val description = if (colon > 0) content.substring(colon + 1).trim() else ""
        val optional = rawName.endsWith("?")
        return SkillParameter(
            name = rawName.removeSuffix("?").trim(),
            description = description,
            required = !optional,
        )
    }

    /** `- read_file(path={{path}}, max_lines=20)` */
    private fun parseStep(content: String): SkillStep {
        val match = STEP.matchEntire(content.trim())
            ?: throw SkillFormatException("step is not tool(arg=value): $content")

        val tool = match.groupValues[1]
        val args = match.groupValues[2]

        val arguments = LinkedHashMap<String, String>()
        splitArguments(args).forEach { pair ->
            val eq = pair.indexOf('=')
            if (eq <= 0) throw SkillFormatException("argument is not name=value: $pair")
            arguments[pair.take(eq).trim()] = pair.substring(eq + 1).trim().trim('"')
        }
        return SkillStep(tool, arguments)
    }

    /** A comma only separates arguments when another `name=` follows it. */
    private val NEXT_ARGUMENT = Regex("""^\s*[A-Za-z_][A-Za-z0-9_]*\s*=""")

    /**
     * Splits an argument list on the commas that actually separate arguments.
     *
     * Not a plain `split(',')`. Values contain commas all the time — a shell
     * command like `git log --pretty=format:%h,%s` is exactly the kind of thing
     * a skill wraps — and splitting there truncates the command instead of
     * failing, which is the worst way to be wrong.
     *
     * So a comma ends an argument only when what follows it starts a new one.
     * Quoting still works for the remaining ambiguity, where a value genuinely
     * contains something shaped like `, name=`.
     */
    private fun splitArguments(text: String): List<String> {
        val out = mutableListOf<String>()
        val current = StringBuilder()
        var quoted = false
        var braces = 0

        text.forEachIndexed { index, c ->
            when {
                c == '"' -> { quoted = !quoted; current.append(c) }
                c == '{' -> { braces++; current.append(c) }
                c == '}' -> { if (braces > 0) braces--; current.append(c) }
                c == ',' && !quoted && braces == 0 &&
                    NEXT_ARGUMENT.containsMatchIn(text.substring(index + 1)) -> {
                    if (current.isNotBlank()) out += current.toString().trim()
                    current.clear()
                }
                else -> current.append(c)
            }
        }
        if (current.isNotBlank()) out += current.toString().trim()
        return out
    }
}

/** Replaces `{{slot}}` references with values; unknown slots are left alone. */
fun applySlots(template: String, values: Map<String, String>): String {
    var out = template
    values.forEach { (key, value) -> out = out.replace("{{$key}}", value) }
    return out
}

/** Slot names a template references, in order of first appearance. */
fun slotsIn(template: String): List<String> =
    Regex("""\{\{([A-Za-z0-9_]+)}}""")
        .findAll(template)
        .map { it.groupValues[1] }
        .distinct()
        .toList()
