package dev.sruti.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The skill format, which users author by hand.
 *
 * Every one of these is a case someone will hit while writing a skill in a text
 * field on a phone, where the only feedback is whether it loads.
 */
class SkillTest {

    private val summarise = """
        ---
        name: summarise-file
        description: Read a file and write a summary beside it
        keywords: summarise, digest
        parameters:
          - path: The file to summarise
          - style?: How terse to be
        steps:
          - read_file(path={{path}})
          - write_file(path={{path}}.summary.md, content={{step1}})
        ---

        Keep facts and numbers; drop repetition.
    """.trimIndent()

    @Test
    fun `parses a complete skill`() {
        val skill = SkillParser.parse(summarise)

        assertEquals("summarise-file", skill.name)
        assertEquals("Read a file and write a summary beside it", skill.description)
        assertEquals(listOf("summarise", "digest"), skill.keywords)
        assertEquals(2, skill.parameters.size)
        assertEquals(2, skill.steps.size)
        assertTrue(skill.isTemplated)
        assertTrue(skill.body.startsWith("Keep facts"))
    }

    @Test
    fun `a trailing question mark marks a parameter optional`() {
        val skill = SkillParser.parse(summarise)
        assertTrue(skill.parameters[0].required)
        assertEquals("path", skill.parameters[0].name)
        assertTrue(!skill.parameters[1].required)
        assertEquals("style", skill.parameters[1].name)
    }

    @Test
    fun `parses step arguments including slot references`() {
        val skill = SkillParser.parse(summarise)

        assertEquals("read_file", skill.steps[0].tool)
        assertEquals(mapOf("path" to "{{path}}"), skill.steps[0].arguments)
        assertEquals(
            mapOf("path" to "{{path}}.summary.md", "content" to "{{step1}}"),
            skill.steps[1].arguments,
        )
    }

    @Test
    fun `a comma followed by another argument does split`() {
        val skill = SkillParser.parse(
            """
            ---
            name: two-args
            description: Two arguments
            steps:
              - write_file(path=/tmp/a.txt, content=hello)
            ---
            """.trimIndent(),
        )
        assertEquals(
            mapOf("path" to "/tmp/a.txt", "content" to "hello"),
            skill.steps[0].arguments,
        )
    }

    @Test
    fun `a comma inside a value does not split the argument`() {
        // Shell commands routinely contain commas, and splitting on them would
        // silently truncate the command rather than fail.
        val skill = SkillParser.parse(
            """
            ---
            name: shell
            description: Run something
            steps:
              - run_shell(command=git log --pretty=format:%h,%s -n 3)
            ---
            """.trimIndent(),
        )
        assertEquals(
            "git log --pretty=format:%h,%s -n 3",
            skill.steps[0].arguments["command"],
        )
    }

    @Test
    fun `a skill with no steps is guidance only`() {
        val skill = SkillParser.parse(
            """
            ---
            name: careful
            description: Answer carefully
            ---

            Think before answering.
            """.trimIndent(),
        )
        assertTrue(!skill.isTemplated)
        assertEquals("Think before answering.", skill.body)
    }

    @Test
    fun `missing frontmatter is rejected with a usable message`() {
        val error = runCatching { SkillParser.parse("name: nope") }.exceptionOrNull()
        assertTrue(error is SkillFormatException)
        assertTrue(error!!.message!!.contains("---"))
    }

    @Test
    fun `a skill without a name or description is rejected`() {
        listOf(
            "---\ndescription: no name here\n---",
            "---\nname: no-description\n---",
        ).forEach { text ->
            assertTrue(runCatching { SkillParser.parse(text) }.exceptionOrNull() is SkillFormatException)
        }
    }

    @Test
    fun `a malformed step names itself in the error`() {
        val error = runCatching {
            SkillParser.parse(
                """
                ---
                name: broken
                description: Broken step
                steps:
                  - read_file path is here
                ---
                """.trimIndent(),
            )
        }.exceptionOrNull()

        assertTrue(error is SkillFormatException)
        assertTrue(error!!.message!!.contains("read_file path is here"))
    }

    @Test
    fun `slot substitution fills known slots and leaves unknown ones`() {
        assertEquals(
            "/notes/a.txt.summary.md",
            applySlots("{{path}}.summary.md", mapOf("path" to "/notes/a.txt")),
        )
        assertEquals("{{missing}}", applySlots("{{missing}}", mapOf("path" to "x")))
    }

    @Test
    fun `slots are reported in order without duplicates`() {
        assertEquals(
            listOf("path", "step1"),
            slotsIn("{{path}} then {{step1}} then {{path}}"),
        )
    }

    // --- matching -------------------------------------------------------------

    private val skills = listOf(
        SkillParser.parse(summarise),
        SkillParser.parse(
            """
            ---
            name: save-page
            description: Fetch a web page and save its text
            keywords: fetch, download, url
            parameters:
              - url: The address
            steps:
              - fetch_url(url={{url}})
            ---
            """.trimIndent(),
        ),
    )

    @Test
    fun `matches the skill whose keywords the request uses`() {
        assertEquals(
            "summarise-file",
            SkillMatcher.match("summarise the file at /notes/a.txt", skills)?.name,
        )
        assertEquals(
            "save-page",
            SkillMatcher.match("download the page at example.com and save it", skills)?.name,
        )
    }

    @Test
    fun `an unrelated request matches nothing`() {
        // Running the wrong skill takes actions the user did not ask for, so a
        // weak match must fall through to the ordinary agent loop.
        assertNull(SkillMatcher.match("what is the capital of France", skills))
        assertNull(SkillMatcher.match("hello", skills))
    }

    @Test
    fun `an empty request matches nothing`() {
        assertNull(SkillMatcher.match("", skills))
        assertNull(SkillMatcher.match("   ", skills))
    }

    @Test
    fun `missing tools are reported against the registry`() {
        val registry = ToolRegistry(emptyList())
        val skill = SkillParser.parse(summarise)
        assertEquals(listOf("read_file", "write_file"), skill.missingTools(registry))
    }
}

/**
 * The skills that ship with the app.
 *
 * They are also the worked examples of the format, so a broken one is both a
 * missing feature and misleading documentation. Extracted from the store's
 * source rather than duplicated, so the test cannot drift from what ships.
 */
class BuiltInSkillTest {

    private fun builtInTexts(): List<String> {
        val source = java.io.File("src/main/java/dev/sruti/agent/SkillStore.kt").readText()
        return Regex("\"\"\"\\s*\\n(\\s*---\\n.*?)\"\"\"", RegexOption.DOT_MATCHES_ALL)
            .findAll(source)
            .map { it.groupValues[1].trimIndent() }
            .toList()
    }

    @Test
    fun `the store declares built-in skills`() {
        assertTrue("no built-in skills found in SkillStore", builtInTexts().size >= 4)
    }

    @Test
    fun `every built-in skill parses`() {
        builtInTexts().forEach { text ->
            val skill = runCatching { SkillParser.parse(text, builtIn = true) }
                .getOrElse { throw AssertionError("built-in skill failed to parse: ${it.message}\n$text") }
            assertTrue(skill.name.isNotBlank())
            assertTrue(skill.description.isNotBlank())
        }
    }

    @Test
    fun `every built-in step names a real tool`() {
        // A skill referring to a tool that does not exist is unreachable, and the
        // only place that shows up is at run time.
        val known = setOf(
            "read_file", "write_file", "list_files", "fetch_url",
            "read_clipboard", "write_clipboard", "run_shell",
        )
        builtInTexts().forEach { text ->
            val skill = SkillParser.parse(text, builtIn = true)
            skill.steps.forEach { step ->
                assertTrue("${skill.name} names unknown tool ${step.tool}", step.tool in known)
            }
        }
    }

    @Test
    fun `every built-in slot resolves to a parameter or an earlier step`() {
        builtInTexts().forEach { text ->
            val skill = SkillParser.parse(text, builtIn = true)
            val declared = skill.parameters.map { it.name }.toMutableSet()

            skill.steps.forEachIndexed { index, step ->
                step.arguments.values.flatMap { slotsIn(it) }.forEach { slot ->
                    assertTrue(
                        "${skill.name} step ${index + 1} uses undefined slot $slot",
                        slot in declared,
                    )
                }
                // Available to every step after this one, never to itself.
                declared += "step${index + 1}"
            }
        }
    }
}
