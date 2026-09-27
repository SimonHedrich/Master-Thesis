# A 1986 Aircraft Manual Fixed My Anti-Slop Skill

*My anti-slop skill already had rules. They worked, and they were entirely mine, invented from a year of irritation and tested against nothing. Replacing that homemade layer with two writing systems that predate the whole AI era, one written so an aircraft mechanic could not misread a maintenance manual and one that runs the largest documentation estate in software, changed what the skill can do. They also contradict each other, which turned out to be the useful part.*

Sascha Becker · July 30, 2026 · 16 min read

> Source: `anti-slop-skill.html` (blog post by Sascha Becker)

---

My anti-slop skill already had rules before any of this. A register to hold, a table of tells with a fix for each, and five rules covering punctuation, structure, vocabulary, stance, and formatting. Drafts came back better with it than without. It was not nothing.

It was, however, entirely mine. I wrote it across a year of irritation at drafts, one phrase at a time, and I had tested it against nothing.

The gap took me a long time to see, because it is structural rather than a missing rule. Almost everything in there described something to remove. Cut the em dash. Drop the hedge. Delete the throat-clearing opener.

Even the vocabulary rule, the one closest to a real instruction, was a list of words to avoid. It carried a warning stapled on: do not dodge a banned word into something more contorted.

> I knew "leverage" would come back as "harness". I had written that down. I simply had no rule that stopped it, because every rule I owned described the output I did not want instead of how to build the sentence I did.

Here is the shape of the problem, from a real README intro a model wrote for a fictional caching library:

> Traditional caches miss constantly in LLM workloads because users rarely phrase the same question identically — fluxcache solves this by embedding incoming prompts and matching them against previously cached queries within a configurable similarity threshold. It ships with sensible defaults so you can get semantic caching running in a few lines of code, while exposing the knobs — similarity thresholds, TTLs, namespacing, custom scoring — that real applications need as they scale. Whether you're building a chatbot, a RAG pipeline, or an agentic workflow, fluxcache is designed to slot into your existing stack with minimal friction and no vendor lock-in.

Quoted verbatim, em dashes and all, which is why that paragraph is the only place on this site you will find one.

My old rules do catch things here. The em dashes are the first row of my tells table, and "whether you're building a chatbot" is the false-inclusivity row. So the skill worked.

Now strip both. Fix the dashes, name the actual audience, and read it again. A 34-word opening sentence. "Sensible defaults." "Minimal friction and no vendor lock-in." "Designed to slot into your existing stack."

Every clause still asserts a benefit that nobody measured, and no rule I had written touched any of it. I had catalogued the phrases that annoyed me and never asked how to build a sentence.

That is the difference between a repair list and a system. A repair list finds only what you already thought to look for.

## The number that changed my mind

Someone actually measured this.[1] A cross-model test ran six realistic writing tasks under four conditions. The tasks were a README, a PR description, API docs, an error message, a getting-started guide, and a deprecation notice. The conditions were a plain baseline, a banned-word list, Orwell's six rules, and a skill built on a controlled language. A heuristic linter then counted violations per 100 words.

On Claude, the banned-word list cut slop by 3 percent. The controlled language cut it by 74 percent.

The honest footnote matters as much as the headline. On GPT the same banned-word list cut 40 percent, so "banning words does nothing" is a result for one model, not a law. The sample is six tasks and two models, scored by a heuristic. It points a direction rather than proving one.

The direction was clear enough. Every condition that handed the model a system beat the one that handed it a list of words to avoid. My own rules were not that list, but they were never in the test either. Nobody had scored them, including me.

## What a controlled language actually constrains

The system is ASD-STE100, Simplified Technical English. The European aerospace industry started work on it in the late 1970s and published the first guide in 1986, for maintenance manuals read by mechanics who often did not speak English first. Getting a sentence wrong on a hydraulic line is not a style problem, so the standard is unusually strict about what a sentence may do.

It is not a museum piece. A maintenance group in Brussels still shepherds it, Issue 9 landed in January 2025, and it reached full international standard status the same year.[2]

The standard has two parts: a set of writing rules, 53 of them in the current issue, and a dictionary of roughly 900 approved words.[3]

The dictionary is the part that surprises people. Each approved word carries exactly one meaning and one part of speech, and the alternatives are simply unavailable:

| Approved | Meaning it is allowed to carry | Not approved |
| --- | --- | --- |
| start | to begin an action | begin, commence, initiate, originate |
| make sure | to confirm a condition | verify, check, confirm, ensure |
| fall | to move down by force of gravity | to decrease |
| about | concerned with | approximately, around |
| check | noun only | the verb "to check" |
| oil | noun only, "the oil is dirty" | the verb, "oil the valve" |

Sit with the "check" row for a second. The word is fine as a noun and forbidden as a verb, because "check the valve" could mean inspect it, test it, or stop it. A mechanic under time pressure should never have to pick. That is the whole philosophy in one entry, and it runs against everything a language model learns, which is to reach for variety.

The sentence rules work the same way. Put the condition before the command, so nobody acts on the first half of a sentence before reaching the second:[4]

> If hot oil touches your skin, injuries can occur.

The rest of the mechanics are equally blunt. Active voice with a named actor, so "the parser reads the file" rather than "the file is read". A verb for an action, so "analyse the log" rather than "perform an analysis of the log". No present perfect, so "we received the reports" rather than "we have received the reports". Descriptive sentences cap at 25 words and procedural ones at 20. One topic per paragraph, six sentences at most.

None of that mentions AI. The standard predates the web, and it served people whose job was to not misunderstand. That is the whole reason it transfers.

Same model, same prompt, same fictional library, with those rules in front of it:

> A normal cache matches requests by exact text. A small change in wording then causes a cache miss. fluxcache compares the meaning of a new prompt with the prompts already in the cache. If two prompts are close enough in meaning, fluxcache returns the stored response instead of a new call to the model. This lowers the number of calls to the model and cuts the cost and response time of the application.

That is the version I would ship. It is shorter, it explains the mechanism instead of asserting a benefit, and it never reaches for "seamless". The rules did not tell it to avoid any of those words. They told it how long a sentence may be and that every sentence needs an actor doing something, and the marketing voice had nowhere left to live.

> **Two modes, because a runbook is not a blog post**
>
> Strict mode applies every rule and expands contractions, which suits procedures, install steps, and error messages. A relaxed mode keeps the sentence, verb, and paragraph discipline but leaves the prose enough range to read naturally. Applying aircraft-manual rules to an essay produces its own kind of unreadable.

## The second guide disagreed with the first

Once one borrowed standard worked, I went looking for another and landed on the Microsoft Writing Style Guide.[5] It runs one of the largest documentation estates in software, and it is public.

Where the aerospace standard is a rulebook, Microsoft teaches by example, and the examples are the useful part. These are theirs, not mine:[6]

| Their before | Their after |
| --- | --- |
| If you're ready to purchase Office 365 for your organization, contact your Microsoft account representative. | Ready to buy? Contact us. |
| Invalid ID | You need an ID that looks like this: [someone@example.com](mailto:someone@example.com) |
| You can access Office apps across your devices, and you get online file storage and sharing. | Store files online, access them from all your devices, and share them with coworkers. |
| Android, iOS and Windows | Android, iOS, and Windows |
| **Move a tile.** | **Move a tile** |

The third row is the one that changed how I edit. Their instruction is "most of the time, start each statement with a verb", and "edit out *you can* when it isn't necessary". Watch what it does: the before sentence has a subject doing nothing ("you can access"), the after sentence has three verbs doing work. The same rule kills the expletive opener, so "there are three ways to configure this" becomes "three options exist, and the default is B".

"There are" and "you can" push the real subject to the back and burn the verb slot on nothing. Cutting them forces you to name the actor, and naming the actor usually exposes vagueness you did not know you had.

The second row is a quiet lesson in error messages. "Invalid ID" is technically accurate and useless. The rewrite tells you what a valid one looks like, which is the only thing you wanted.

Microsoft also says "if you mean the same thing, use the same word". That is the aerospace one-name rule, reached from a completely different argument. Finding the same rule twice from two directions is a good sign you have found a real one.

Then I hit the part I did not plan for.

Microsoft's tip number nine is called "Don't be spacey", and it is about dash spacing. Their recommended version reads:

> Use pipelines—logical groups of activities—to consolidate activities that are part of a task.

That is the single loudest generated-text marker there is, published as house style. Tip number three is "Project friendliness", which requires contractions like *it's*, *you'll*, and *we're*. The brand voice page asks for "warm and relaxed", "less head, more heart", and says "occasionally, we're fun".[7]

My skill kills all three on sight.

## Resolving a conflict beats averaging it

My first instinct was to take the parts I agreed with and quietly drop the rest. That is how you end up with a rulebook nobody can apply. The reasoning is missing, so the next person cannot tell a real exception from an oversight.

So the conflicts got written down as conflicts, with a verdict on each and the reason behind it. Microsoft writes consumer product UI for a global audience, where a dialog box has to sound reassuring. I write for a reader who wants the fact and finds decoration irritating. Neither of us is wrong. We have different readers.

The verdict is that my register wins by default, and a third mode exists for when it should not. If the text ships into a surface that already follows Microsoft style, the house wins and all three inversions apply. Contractions come back, em dashes are fine, the voice warms up. Every other rule stays in force, because a marketing adjective is slop in anyone's house.

That mode only exists because I wrote the disagreement down instead of smoothing it over.

## The part I did not expect

The same guide carries a large section on accessibility and inclusive language, and reading it showed me a gap I had no rule for.

Consider "the button above". It is not slop. It is plain, short, and direct, and it passes every rule I had. It is also useless to someone whose screen reader moves through the page in a line, and it breaks again when the layout reflows on a phone.

| Before | After |
| --- | --- |
| Click the button above to retry. | Select **Retry** on the toolbar. |
| For more details, [click here](#). | Read the [retry policy](#). |
| See the table below for the supported flags. | The following table lists the supported flags. |
| Press the green button. | Select **Deploy**. |

The second row is the one that shows up everywhere. Assistive software can list a page's links out of context, so a page full of "click here" becomes a list of identical entries with no way to tell them apart. The last row fails for anyone who cannot distinguish the colour, which is why the fix names the button instead of describing it.

Their reasoning on directional words is worth quoting, because it is more specific than "be accessible". Left, right, up, down, above, and below "aren't very useful for people who use screen-reading software", so the guidance is to write "the first item in the following list" or "on the toolbar" instead.[8] The same section says to avoid verbs like *click* and *swipe*, because they assume a mouse or a touchscreen and exclude every other input method. It even asks you to spell out *and*, *plus*, and *about* rather than using `&`, `+`, and `~`, since screen readers misread or skip them.

The bias-free section is equally concrete. It is a substitution table, not a lecture:[9]

| Use this | Not this |
| --- | --- |
| chair, moderator | chairman |
| workforce, staff, personnel | manpower |
| humanity, people | mankind |
| primary/subordinate | master/slave |
| perimeter network | demilitarized zone, DMZ |
| stop responding | hang |

And on the generic "he", their own before-and-after: "If the user has the appropriate rights, he can set other users' passwords" becomes "If you have the appropriate rights, you can set other users' passwords". Second person solves it without any awkwardness, which is the point. Rewriting is easier than the "he/she" constructions people reach for.

It goes further than my table does. Put the person before the disability, and drop words that imply pity, like "stricken with" or "suffering from". Do not mention a disability at all unless it is relevant.

None of this reads as generated. All of it is a defect, and an agent producing documentation at volume introduces it faster than a person would. The material has nothing to do with AI slop and belonged in the skill anyway.

## Making it checkable

Rules an agent can ignore are suggestions. So I wrote a small linter for the mechanical subset, and it prints violations per 100 words.

Here it is on the baseline README from that experiment, the one a model wrote with no writing instruction at all:

```text
$ node slop-lint.mjs baseline.md

baseline.md  words=97  violations=8  score=8.20/100w  longest=35w  (flavored)
  em_dash                      3  L1 "—", L1 "—", L1 "—"
                                  -> period, comma, colon, or parentheses
  long_sentence(>25w)          3  L1 "34 words", L1 "35 words", L1 "28 words"
                                  -> split it, one idea per sentence
  false_inclusivity            1  L1 "Whether you're"
                                  -> say who it is actually for
  passive_voice                1  L1 "is designed"
                                  -> name the actor and make it active
```

Every hit carries a line number, the matched text, and the fix. Same prompt, same model, with the writing system applied:

```text
$ node slop-lint.mjs ste.md

ste.md  words=73  violations=0  score=0.00/100w  longest=25w  (flavored)
```

Zero violations and 25 percent fewer words, saying the same thing.

## The score lies if you read it as a grade

Only the delta means anything. Lint, revise, lint again, and the drop tells you something. The absolute number does not, and my own repo proves it:

```text
$ node slop-lint.mjs references/*.md

inclusive-and-accessible.md  words=417   violations=0   score=0.00/100w
microsoft-style.md           words=458   violations=0   score=0.00/100w
ste.md                       words=1441  violations=20  score=1.40/100w
prose.md                     words=1194  violations=74  score=6.20/100w
```

That last file is the worst-scoring document I have, and it is correct. `prose.md` is the catalogue of bad writing, a page of before-and-after pairs. Every "before" is slop on purpose. A linter cannot tell quotation from confession.

The same caveat applies at the short end. Under about fifty words, one hit reads as ten per hundred, so a two-sentence deprecation notice scores like a disaster.

Running it on my own files was educational anyway. It found a semicolon in a document two sections below the rule banning semicolons. While writing the accessibility rules, I used the phrase "the three conflicts above" in the same commit that added the rule against directional references. Both were mine, both were minutes old, and reading would not have caught either.

## What the borrowing actually bought

The skill is better, but the more useful lesson is about where the rules came from.

I invented rules from my own irritation, one phrase at a time. People with real stakes argued over the borrowed standards for decades: a mechanic who cannot misread a procedure, a documentation team serving a hundred locales. They arrive tested, they explain themselves, and they cover cases I would never have reached alone. The accessibility rules are the clearest example. Irritation at the word "delve" never produces them.

The contradiction between them turned out to be worth as much as the agreement. Two credible systems disagreeing on the em dash forced me to say out loud who I write for, which is a thing I had never actually written down.

## The limit

All of it fixes the form of slop, never the substance. A text can score zero and still say nothing at all. The rules will make a hollow paragraph shorter, plainer, and easier to read, and it will still be hollow, because no linter knows whether a claim is true or worth making.

That part is still my job. It just got easier to see, now that the noise around it is gone.

> **There's a skill for this**
>
> The construction rules, both style guides with the conflicts named, the accessibility and inclusive-language reference, and the linter live as an agent skill. Install it with `npx skills@latest add saschb2b/skills --skill no-slop` or read the full [no-slop skill](/skills/no-slop).

The reference material behind this skill ships as an OKF bundle, the format for packaging knowledge an agent reads on demand instead of dumping it all into one file.

## Sources

- [The cure for AI slop is a 1986 aircraft manual](https://github.com/woosal1337/blog/tree/main/videos/ep01-the-cure-for-ai-slop) — The cross-model experiment behind the 3 percent and 74 percent numbers, with the raw outputs and the scoring script.
- [ASD-STE100 Simplified Technical English](https://asd-ste100.org) — The controlled language itself, Issue 9 since January 2025. Free to download after a short request form.
- [ASD-STE100 FAQ](https://asd-ste100.org/STE_faq.html) — The clearest short explanation of the two parts, the roughly 900-word dictionary, and why one word gets one meaning.
- [TechScribe, ASD Simplified Technical English](https://www.techscribe.co.uk/techw/asd-simplified-technical-english.htm) — Worked before-and-after examples from a technical writer who uses the standard professionally.
- [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/) — Public, thorough, and opinionated. The accessibility and bias-free sections are the parts most style guides skip.
- [Top 10 tips for Microsoft style and voice](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice) — The condensed version, and the source of the before-and-after pairs quoted here. Also where the em dash recommendation lives.
- [Writing for all abilities](https://learn.microsoft.com/en-us/style-guide/accessibility/writing-all-abilities) — Why directional words and click-here links break for screen readers, with the specific replacements.
- [saschb2b/skills](https://github.com/saschb2b/skills) — The open skills repo. no-slop lives under productivity, with its references and the linter.
