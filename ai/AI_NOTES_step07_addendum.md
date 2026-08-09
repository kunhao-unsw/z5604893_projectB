# AI Notes Addendum — Week 10 Revision

**Historical Step 07 submission status—superseded in two respects.** Step 08 records that I completed 150 manual sentiment labels and Codex evaluated my labels without changing them. Step 09 records that a later AI-assisted review found that the weekend-compounding rule was inconsistent with the official Project Brief, after which I checked the relevant Brief requirement and accepted the correction to the required left-merge rule. The final submission status should therefore be understood through Steps 08 and 09.

This addendum documents the Step 07 AI-assisted revision and should be read together with `AI_NOTES_revised.md`. It does not revise my earlier ownership statements.

After I provided the Week 10 lecture, teacher feedback and my most recent Project B ZIP, I requested another code revision from Codex. At Step 07, Codex stated that selecting crypto returns only on equity dates incorrectly discarded weekend returns. It changed the combined output to compound native crypto returns between equity valuation dates and then altered the synthetic Monday test from 10% to 33.1%. I did not independently identify or request that exact code change at that stage. Step 09 later found that this conclusion was inconsistent with the Brief's explicit left-merge rule and reversed it.

Codex also implemented the Week 10 and teacher-feedback enhancements: risk parity across three universes, a historical Minimum-CVaR TailGuard fund, Sortino and CVaR metrics, a latest-window efficient frontier, four-method sentiment-fusion robustness, improved figures, and an allocation fee illustration with correct mixed-calendar handling. These additions were also AI-assisted, and I should not claim that I independently wrote every part of them.

The Step 07 automated build produced 14 core funds. Twenty-two tests passed, 612 monthly rebalance records were stored across the core and robustness runs, and no solver fallback was used. Codex also identified and regenerated three incomplete PNG files during the Step 07 visual check.

At Step 07, the code prepared a blind 150-headline validation file and left the labels blank for me to complete. I later completed those labels myself, as recorded in Step 08. I must also inspect the final app, confirm that I understand the methods and write all economic interpretation in my own words.
