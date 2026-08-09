# Part B Prompt Log 01 — Initial Review and Return-Calendar Correction

## What I wanted

I had already written a first version of Project B. I did not want the AI to start from an unrelated solution. I wanted Codex to review my existing files against the official Project Brief and the Week 7–9 lecture material, preserve my original version, and explain the important problems before revising it.

## My prompt summaries

The bullets summarise the request. The block quote below is an English translation of my original Chinese prompt, not a verbatim English transcript.

> Project A complete and Project B first version written.  
> Please review my existing code against the Project Brief and the Week 7–9  
> lectures, explain what needs to change, and preserve my original version.

I explained that I had already written a first version of Project B and wanted Codex to review that existing work, not to start from a blank project.

I asked Codex to compare my first version against the official Project Brief and the Week 7–9 lecture material.

I asked Codex to identify the most important problems first, explain why they mattered, and revise the project step by step.

## What Codex identified

Codex compared my first version with the brief and identified two important problems in the return and performance code:

My version reindexed crypto prices to equity dates before calculating returns. That would drop weekend crypto prices and make the return for Monday a combination of the whole Friday-to-Monday movement.

My first performance function calculated the Sharpe ratio as CAGR divided by annualised volatility rather than average daily excess return divided by daily volatility.

I did not independently identify those two bugs before the audit. My input here was providing my existing implementation, the official brief and the lecture material, and asking for a rule-by-rule review.

## What Codex changed

It replaced the single mixed return workflow with separate equity, crypto and combined return panels.

It calculated crypto returns on the crypto 365-day calendar first, and only then aligned the already-calculated returns to the equity calendar for combined funds.

It kept the standalone 365-day crypto return panel for crypto-only funds.

It added separate 252- and 365-day annualisation constants.

It changed the Sharpe ratio to use mean daily excess return divided by daily volatility and then annualise that ratio.

## What was wrong or risky

The original calendar order wasn't a minor formatting problem. Comparing the old and revised return panels showed 213 changed equity-calendar observations for each of the 10 crypto assets. The original Sharpe calculation was also risky since it mixed a geometric return measure (CAGR) with a volatility measure when the required ex-post Sharpe ratio is based on mean periodic excess return.

## Evidence produced by Codex

A synthetic-price test confirmed that the Monday crypto return is now the Sunday-to-Monday return rather than a Friday-to-Monday cumulative return.

A direct formula test checked the revised Sharpe calculation.

A future-data mutation test checked that changing observations after the first rebalance doesn't change the first rebalance weights.

These three focused tests passed at this checkpoint.

## My review status

Codex wrote and ran the focused tests. I reviewed the explanations, checked why the changes were necessary, and asked it to continue the full revision. Before submission, I still need to run the final test suite locally and confirm that I can explain why returns must be calculated within each asset panel before calendar alignment.
