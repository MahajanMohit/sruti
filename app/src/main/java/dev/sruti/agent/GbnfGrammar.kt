package dev.sruti.agent

/**
 * Builds GBNF grammars that constrain what the model is allowed to emit.
 *
 * This is the load-bearing decision of the whole agent. A 1–2B model asked to
 * produce JSON will produce *something like* JSON: a trailing comma, an unquoted
 * key, a hallucinated field, a chatty preamble before the brace. Every one of
 * those is a failed tool call, and prompting harder does not fix it at this size.
 *
 * A grammar removes the failure mode instead of reducing it. Tokens that would
 * break the structure are masked out during sampling, so a malformed call is not
 * unlikely — it is unreachable. What remains is whether the model chose sensibly,
 * which is a different and much more tractable problem.
 *
 * Deliberately split into two narrow questions rather than one wide one. Asking a
 * small model to pick a tool *and* fill its arguments in a single pass is markedly
 * less reliable than asking it to pick, then asking it to fill.
 */
object GbnfGrammar {

    /** Emitted when the model judges that no tool applies. */
    const val NO_TOOL = "none"

    /**
     * A grammar accepting exactly one tool name, or [NO_TOOL].
     *
     * The escape hatch is not optional. Without it the model must name a tool on
     * every turn, and a model forced to choose will choose — turning "I can't help
     * with that" into a confidently wrong action.
     */
    fun toolChoice(tools: List<Tool>): String {
        val names = (tools.map { it.name } + NO_TOOL).distinct()
        // Each alternative is a single quoted literal containing escaped quotes:
        // "\"read_file\"". Splitting it into three tokens would leave the name as
        // a bare identifier, which GBNF reads as a reference to an undefined rule
        // and rejects the entire grammar.
        val alternatives = names.joinToString(" | ") { "\"\\\"" + escape(it) + "\\\"\"" }
        return "root ::= $alternatives\n"
    }

    /**
     * A grammar accepting exactly one well-formed argument object for [tool].
     *
     * Required parameters appear in a fixed order; optional ones follow, each
     * individually skippable. Fixing the order costs nothing — the model is
     * filling slots, not composing — and makes the grammar unambiguous.
     */
    fun toolCall(tool: Tool): String {
        val required = tool.parameters.filter { it.required }
        val optional = tool.parameters.filterNot { it.required }

        val builder = StringBuilder()
        val usedTypes = linkedSetOf<ToolParameter.Type>()

        if (tool.parameters.isEmpty()) {
            builder.append("root ::= \"{\" ws \"}\"\n")
            builder.append(wsRule())
            return builder.toString()
        }

        val parts = mutableListOf<String>()
        required.forEachIndexed { index, parameter ->
            if (index > 0) parts += "\",\" ws"
            parts += entry(parameter, usedTypes)
        }

        optional.forEach { parameter ->
            val separator = if (required.isEmpty() && parts.isEmpty()) "" else "\",\" ws "
            // Each optional entry is independently skippable, so any subset in
            // declaration order is accepted.
            parts += "( $separator${entry(parameter, usedTypes)} )?"
        }

        builder.append("root ::= \"{\" ws ")
        builder.append(parts.joinToString(" "))
        builder.append(" \"}\"\n")

        usedTypes.forEach { type -> builder.append(ruleFor(type)) }
        builder.append(wsRule())
        return builder.toString()
    }

    /** `"key" ws ":" ws <value>` for one parameter. */
    private fun entry(
        parameter: ToolParameter,
        usedTypes: MutableSet<ToolParameter.Type>,
    ): String {
        val key = "\"\\\"" + escape(parameter.name) + "\\\"\""
        val value = when (parameter.type) {
            ToolParameter.Type.Enum -> {
                // Inlined rather than named: enum values are per-parameter, so a
                // shared rule would collide between two enum parameters.
                if (parameter.values.isEmpty()) {
                    usedTypes += ToolParameter.Type.String
                    "string"
                } else {
                    "( " + parameter.values.joinToString(" | ") {
                        "\"\\\"" + escape(it) + "\\\"\""
                    } + " )"
                }
            }
            else -> {
                usedTypes += parameter.type
                ruleName(parameter.type)
            }
        }
        return "$key ws \":\" ws $value ws"
    }

    private fun ruleName(type: ToolParameter.Type): String = when (type) {
        ToolParameter.Type.String -> "string"
        ToolParameter.Type.Integer -> "integer"
        ToolParameter.Type.Number -> "number"
        ToolParameter.Type.Boolean -> "boolean"
        ToolParameter.Type.Enum -> "string"
    }

    private fun ruleFor(type: ToolParameter.Type): String = when (type) {
        // Only the escapes JSON actually defines; anything else would let the
        // model emit a string that fails to parse afterwards.
        ToolParameter.Type.String, ToolParameter.Type.Enum ->
            """string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""""" + "\n"

        ToolParameter.Type.Integer ->
            """integer ::= "-"? ( "0" | [1-9] [0-9]* )""" + "\n"

        ToolParameter.Type.Number ->
            """number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?""" + "\n"

        ToolParameter.Type.Boolean ->
            """boolean ::= "true" | "false"""" + "\n"
    }

    // Bounded rather than [ \t\n]*: an unbounded whitespace rule lets the model
    // emit indefinite padding and never terminate.
    private fun wsRule(): String = "ws ::= [ \\t\\n]{0,4}\n"

    /** Escapes a literal for inclusion in a GBNF double-quoted string. */
    private fun escape(text: String): String = buildString {
        text.forEach { c ->
            when (c) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(c)
            }
        }
    }
}
