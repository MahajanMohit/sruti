package dev.sruti.agent

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Tests for grammar generation.
 *
 * These assert structure, which is necessary but not sufficient: a grammar can be
 * structurally plausible and still be rejected by llama.cpp's parser, or accepted
 * and fail to constrain anything. So each generated grammar is also written to
 * `build/grammars/`, where `tools/verify_grammars.sh` feeds it to a real model and
 * checks that the output actually conforms.
 */
class GbnfGrammarTest {

    private val readFile = Tool(
        name = "read_file",
        description = "Read a text file from the app's storage",
        parameters = listOf(
            ToolParameter("path", ToolParameter.Type.String, "File path"),
            ToolParameter("max_lines", ToolParameter.Type.Integer, "Line limit", required = false),
        ),
        execute = { ToolResult.Success("") },
    )

    private val setMode = Tool(
        name = "set_mode",
        description = "Change the mode",
        parameters = listOf(
            ToolParameter(
                name = "mode",
                type = ToolParameter.Type.Enum,
                description = "Which mode",
                values = listOf("fast", "balanced", "quality"),
            ),
        ),
        execute = { ToolResult.Success("") },
    )

    private val noArgs = Tool(
        name = "get_time",
        description = "Current time",
        execute = { ToolResult.Success("") },
    )

    private fun emit(name: String, grammar: String): String {
        val dir = File("build/grammars").apply { mkdirs() }
        File(dir, "$name.gbnf").writeText(grammar)
        return grammar
    }

    // --- tool choice ---------------------------------------------------------

    @Test
    fun `tool choice grammar lists every tool as a quoted alternative`() {
        val grammar = emit("choice", GbnfGrammar.toolChoice(listOf(readFile, setMode, noArgs)))

        assertTrue(grammar.startsWith("root ::="))
        listOf("read_file", "set_mode", "get_time").forEach { name ->
            assertTrue("expected $name in:\n$grammar", grammar.contains("\\\"$name\\\""))
        }
    }

    @Test
    fun `tool choice always offers an escape hatch`() {
        // Without it the model must name a tool every turn, which turns "I cannot
        // help with that" into a confidently wrong action.
        val grammar = GbnfGrammar.toolChoice(listOf(readFile))
        assertTrue(grammar.contains("\\\"${GbnfGrammar.NO_TOOL}\\\""))
    }

    @Test
    fun `tool choice with no tools still yields a valid grammar`() {
        val grammar = GbnfGrammar.toolChoice(emptyList())
        assertTrue(grammar.startsWith("root ::="))
        assertTrue(grammar.contains(GbnfGrammar.NO_TOOL))
    }

    // --- tool call -----------------------------------------------------------

    @Test
    fun `tool call grammar requires required parameters and makes optional ones skippable`() {
        val grammar = emit("read_file", GbnfGrammar.toolCall(readFile))

        assertTrue(grammar.contains("\\\"path\\\""))
        assertTrue(grammar.contains("\\\"max_lines\\\""))
        // The optional entry is wrapped in ( ... )? so any subset is accepted.
        assertTrue("optional parameter must be skippable:\n$grammar", grammar.contains(")?"))
        assertTrue(grammar.contains("string ::="))
        assertTrue(grammar.contains("integer ::="))
    }

    @Test
    fun `enum parameters are constrained to their declared values`() {
        val grammar = emit("set_mode", GbnfGrammar.toolCall(setMode))

        listOf("fast", "balanced", "quality").forEach { value ->
            assertTrue("expected $value in:\n$grammar", grammar.contains("\\\"$value\\\""))
        }
        // An enum with values must not fall back to a free string, which would
        // defeat the point of constraining it.
        assertFalse(grammar.contains("string ::="))
    }

    @Test
    fun `enum with no values degrades to a string rather than an empty alternation`() {
        // An empty alternation would be a parse error, taking down the whole call.
        val tool = setMode.copy(
            parameters = listOf(
                ToolParameter("mode", ToolParameter.Type.Enum, "Which mode", values = emptyList()),
            ),
        )
        val grammar = GbnfGrammar.toolCall(tool)
        assertTrue(grammar.contains("string ::="))
    }

    @Test
    fun `parameterless tool accepts only an empty object`() {
        val grammar = emit("get_time", GbnfGrammar.toolCall(noArgs))
        assertTrue(grammar.contains("root ::= \"{\" ws \"}\""))
    }

    @Test
    fun `every referenced rule is defined`() {
        val tool = Tool(
            name = "everything",
            description = "All parameter types",
            parameters = listOf(
                ToolParameter("s", ToolParameter.Type.String, "s"),
                ToolParameter("i", ToolParameter.Type.Integer, "i"),
                ToolParameter("n", ToolParameter.Type.Number, "n"),
                ToolParameter("b", ToolParameter.Type.Boolean, "b"),
            ),
            execute = { ToolResult.Success("") },
        )
        val grammar = emit("everything", GbnfGrammar.toolCall(tool))

        val defined = Regex("^(\\w+) ::=", RegexOption.MULTILINE)
            .findAll(grammar).map { it.groupValues[1] }.toSet()

        // A grammar referencing an undefined rule is rejected wholesale by
        // llama.cpp, so this is the difference between working and not.
        listOf("root", "string", "integer", "number", "boolean", "ws").forEach { rule ->
            assertTrue("rule '$rule' referenced but not defined:\n$grammar", rule in defined)
        }
    }

    @Test
    fun `no rule is defined twice`() {
        // Two parameters of the same type must share one rule; a duplicate
        // definition is a parse error.
        val tool = readFile.copy(
            parameters = listOf(
                ToolParameter("a", ToolParameter.Type.String, "a"),
                ToolParameter("b", ToolParameter.Type.String, "b"),
                ToolParameter("c", ToolParameter.Type.Integer, "c"),
                ToolParameter("d", ToolParameter.Type.Integer, "d"),
            ),
        )
        val grammar = GbnfGrammar.toolCall(tool)

        val names = Regex("^(\\w+) ::=", RegexOption.MULTILINE)
            .findAll(grammar).map { it.groupValues[1] }.toList()
        assertTrue("duplicate rule definitions in:\n$grammar", names.size == names.toSet().size)
    }

    @Test
    fun `whitespace is bounded`() {
        // An unbounded whitespace rule lets the model emit padding forever and
        // never reach a terminating brace.
        val grammar = GbnfGrammar.toolCall(readFile)
        assertTrue("whitespace must be bounded:\n$grammar", grammar.contains("ws ::= [ \\t\\n]{0,4}"))
        assertFalse(grammar.contains("ws ::= [ \\t\\n]*"))
    }

    @Test
    fun `literals containing quotes and backslashes are escaped`() {
        val tool = setMode.copy(
            parameters = listOf(
                ToolParameter(
                    name = "weird\"name",
                    type = ToolParameter.Type.Enum,
                    description = "d",
                    values = listOf("a\"b", "c\\d"),
                ),
            ),
        )
        val grammar = GbnfGrammar.toolCall(tool)

        // Unescaped, either of these would terminate the GBNF literal early and
        // produce a grammar that fails to parse.
        assertTrue(grammar.contains("weird\\\"name"))
        assertTrue(grammar.contains("a\\\"b"))
        assertTrue(grammar.contains("c\\\\d"))
    }

    @Test
    fun `string rule permits only the escapes JSON defines`() {
        val grammar = GbnfGrammar.toolCall(readFile)
        // Allowing arbitrary escapes would let the model emit a string that
        // constrains cleanly and then fails to parse as JSON afterwards.
        assertTrue(grammar.contains("""["\\/bfnrt]"""))
    }
}
