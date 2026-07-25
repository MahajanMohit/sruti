package dev.sruti.agent

/**
 * Picks the skill that matches a request, or none.
 *
 * Keyword scoring rather than asking the model, and deliberately so. Choosing
 * between skills is the same kind of question as choosing between tools, and
 * that is the one measured to be unreliable at this model size — a 0.5B model
 * returns a valid-but-wrong name often enough that letting it decide would make
 * skills less predictable than not having them.
 *
 * The threshold matters as much as the ranking. Running a wrong skill is worse
 * than running none, because a skill takes actions without asking which tool to
 * use, so a weak match falls back to the ordinary agent loop.
 */
object SkillMatcher {

    /** Below this, no skill is considered a match. */
    private const val MINIMUM_SCORE = 4

    fun match(request: String, skills: List<Skill>): Skill? {
        val terms = tokenize(request)
        if (terms.isEmpty()) return null

        return skills
            .map { skill -> skill to score(skill, terms) }
            .filter { it.second >= MINIMUM_SCORE }
            .maxWithOrNull(
                compareBy<Pair<Skill, Int>> { it.second }.thenByDescending { it.first.name },
            )
            ?.first
    }

    /** Exposed for the skills screen, which shows why a skill would be chosen. */
    fun score(skill: Skill, terms: Set<String>): Int {
        val nameTerms = tokenize(skill.name)
        val keywordTerms = skill.keywords.flatMap { tokenize(it) }.toSet()
        val descriptionTerms = tokenize(skill.description)

        var score = 0
        terms.forEach { term ->
            if (term in nameTerms) score += 5
            if (term in keywordTerms) score += 3
            if (term in descriptionTerms) score += 1
        }
        return score
    }

    fun tokenize(text: String): Set<String> =
        text.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length > 2 && it !in STOP_WORDS }
            .toSet()

    // Common enough to match everything, which would make every score equal.
    private val STOP_WORDS = setOf(
        "the", "and", "for", "with", "from", "into", "please", "can", "you",
        "this", "that", "then", "get", "use", "using",
    )
}
