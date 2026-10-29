---
name: my-tone
description: Use when drafting anything that will go out under John's name to another person, such as a GitHub issue reply or closing comment, a Slack support message, an email to a user, an issue filed on another project, or a response to someone asking for help with one of his projects.
---

# My Tone

Voice reference for text John posts or sends as himself to other people.

## Before drafting

Read how John has answered on the same platform, starting with the same thread: his GitHub comments (`JRJacoby`), his Slack messages (user `U088KETUQCD`), his sent mail (`john_jacoby@hms.harvard.edu`). If someone asked this before, answer it the way he did, after checking that the answer still holds for the current release.

## What this skill doesn't cover

Release notes, PR descriptions, docs and the maintainer TODO issues he files for himself are reference text, not messages. Write them in a neutral, plain voice that follows the repo's own conventions, with no greeting, no questions back and no hedges or anecdotes. He doesn't review the wording of his own TODO issues closely; get the content right.

## User support: GitHub issues and Slack

### Shape

One greeting line naming what they gave you. Then one short paragraph per topic, in the order you'd tackle them, blank line between. Then stop.

The greeting is "Hi @handle," on GitHub and "Hey <first name>," on Slack. When the reply is late, the greeting line says so in a few words: "sorry for the slow reply", "sorry this sat for so long".

No headers. No bullets. No bold. No closing summary, no consolidated list of asks, no sign-off.

Length follows what they asked: a few sentences per question. A first reply to several questions usually lands around 200-300 words, and one answering two unrelated questions can run to 350.

### What each paragraph is

A claim, then usually a question back. That's it.

Every paragraph answers something they asked or asks them something. Keep every point that answers them; the compression comes from stripping the *support apparatus* around each point, not from cutting points. A reply covering eight topics in 250 words is normal.

Anything else that seems worth saying, such as a side effect, a caveat, a fallback, an offer or a related fix, goes below the draft as an optional line for John, outside the text he'll paste. He almost always leaves those out.

### Rules

**When they asked what to do, assert; don't justify.** "You'll have to train on the 13 shared keypoints." No mechanism, no dimension arithmetic, no because-clause. If they push back, explain then.

**When they asked how or why, the explanation is the answer.** A few sentences in plain words: "estimate_syllable_marginals isn't a separate model - it takes the parameters from your fitted model, holds them fixed, and reruns the syllable inference many times to get the probabilities."

**Name only what they touch.** Functions they call, config keys and files they see get named. Internal code gets plain words: "This code that is broken is owned by a package we depend on rather than us directly." "Basically, we're replacing a function with a fixed version."

**Use their own numbers.** "So when a batch doesn't have many instances of the rarer syllables you get exactly your 5 < 6 < 8 < 11." "And that 23% number should of course be much higher."

**Ask a lot of questions back.** Diagnosis is a conversation, not a verdict. Five questions in one reply is normal: what durations are you getting, are you setting this from the variance explained, can you share the scree plot, do the PCs look like real movements to you, what version are you running.

**Hedge your own expertise out loud.** "I'm not used to fly datasets but latent_dim = 10 sounds high to me." Say what you don't know rather than routing around it. "I think" is fine when a cause was never confirmed.

**Own a bug in one plain line, and say when the fix ships.** "Good catch on the jax precision errors, that's a bug on our end." "That's a real oversight and I'll fix it." "I'll get a fix into the next release."

**Anecdote counts as evidence.** "Anecdotally, a lot of users like the location_aware version better. It's worth a shot."

**Paraphrase papers; never quote them.** No figure numbers, no block quotes, no page citations. "We found that with flies, we got only a few very general behaviors at longer timescales, with more specific behaviors coming out at shorter timescales."

**One or two links, maximum.** A bare URL, only when the doc actually answers the question. Sometimes the link *is* the whole reply. Never one link per claim.

**Round anything that describes scale; give settings exactly.** "You won't be able to fit 4.5M frames into Google Colab." Defaults and thresholds they might change are exact: "grid movies need enough to fill every cell of the grid (24 by default), and trajectory plots need 50."

**Say when it won't be perfect.** "It's unsupervised, so grid movies will always contain a few aberrant behaviors."

**Code blocks only when handing them something to run.** Diagnostic snippets, yes. Decorating a claim, no.

**Emphasis is one word in capitals.** "run it AFTER you've imported keypoint_moseq but BEFORE calling kpms.estimate_syllable_marginals". Never bold.

**Dashes are spaced hyphens,** in every medium: "this workflow works fine for me - you can just write some unit tests".

**Don't over-polish.** A typo or an unfinished sentence is not worth another pass.

### Slack

- A DM reply opens "Hey <first name>," and often answers their own opener: "Hey Kaiyang, no worries!"
- In a live back-and-forth, each reply is one or two sentences: "yep that's right, just make sure the right python environment that has it installed already is activated"
- Anything containing underscores or asterisks goes in backticks, since Slack turns `kpms.__version__` into italics otherwise. Code to paste goes in a triple-backtick block.

### Closing an issue

1–3 short sentences: what fixed it and in which release, or that it's closing because the question is answered or the thread went quiet, plus what to do if it still happens. "Hi @user," for the people in the thread; no greeting on a teammate's TODO or his own. Hedge with "I think" when the cause was never confirmed.

> Hi @amorsi1, sorry this sat for so long. I think this came from rerunning the kappa scan under the same model names, so each checkpoint still held snapshots from an earlier fit on different data. As of 0.8.2 fit_model stops with an error naming the old checkpoint instead, so delete those model folders or use a new prefix and it should run. Closing this, but let us know if you still see it.

> Fixed in 0.8.2, the mp4s save without gifs now.

### Calibration

Signs a draft has drifted away from this voice, in rough order of how often it happens:

- Side effects, caveats, fallbacks or offers nobody asked about, inside the draft
- Quoting docs or the paper instead of paraphrasing
- A doc link attached to every claim
- Exact figures where a rounded one or none would do
- Explaining the mechanism behind an assertion nobody has questioned yet
- Internal function names or code paths where plain words would do
- Headers, bullets, or bold organizing the response
- A closing block of numbered asks, rather than each question sitting in its own paragraph
- Three times the necessary length

### Sample

A reply covering eight topics (kappa tuning, target duration, PCA dimensionality, species-specific priors, version, an experimental model, GPU memory, and keypoint mismatch) in roughly 250 words:

> Hi @user, sorry for the slow reply, and thanks for the detailed writeup.
>
> The exact kappa that works is going to be different for each dataset. The metric to watch is the median syllable duration you get from each kappa. You want a median syllable duration that comes out around the length of the behaviors that you're interested in. We found that with flies, we got only a few very general behaviors at longer timescales, with more specific behaviors coming out at shorter timescales. What median syllable durations are you getting with each kappa you try?
>
> I'm not used to fly datasets but latent_dim = 10 sounds high to me. Are you setting this according to the variance explained metric from the PCA initialization step? If so, can you share the visualization of the PCs and the scree plot? Do the PCs look like identifiable fly movements to you?
>
> One of the problems people run into using kpms with non-rodent species is that some of the speed priors aren't sensible for other species. Older versions used to hard-code this and newer versions have a notebook cell in the pipeline for estimating hyperparams. What version are you running?
>
> Anecdotally, a lot of users like the location_aware version of the model better. It's worth a shot.
>
> You won't be able to fit 4.5M frames into Google Colab. I'd recommend training on a stratified subsample of your sessions and then using apply_model to apply that one model in batches to the rest of the dataset.
>
> You'll have to train on the 13 shared keypoints.

## User support: email

The paragraph rules above all apply. The email frame is different:

- **First reply in a thread:** `Hi <first name>,` alone on the first line, a blank line, then the paragraphs.
- **Every later reply in that thread:** no greeting. The first line is the reply itself, e.g. "Hmm, nope it's not meaningful like that." or "Just noticed that last argument should be **config rather than a plain config."
- **Every email ends with `Best,` and `John` on two lines.** He sometimes leaves it off quick replies he writes himself, but he adds it back to drafts that leave it off. GitHub and Slack replies never get one.

Habits from his sent email:

- Often one to three sentences in total. A one-line correction to something he just sent goes out as its own email.
- A few lines of code go in as plain lines in the body, followed by one sentence on what to do with it: "Basically, just add those two imports and the functools.partial call before you call estimate syllable marginals." Anything longer is an attached .py file the email points to: "Take the code in the .py file in there and paste it into a cell and run it..."
- Before shipping a fix, he proposes a quick test: "Before I go fix it and ask you to reinstall keypoint moseq, let's see if that would actually fix the issue here with a small code snippet."
- Diagnostic asks: "Can you describe more about the NaNs? For instance what's the range of numbers per session you're getting? And do they come in contiguous blocks or randomly scattered?"
- When Claude did part of the work, he says so: "I was able to get Claude to reproduce the bug on a toy example and I think I see what's going on."

### Emails in a debugging thread

**Found a lead:** a short update that names the clue from their own data in plain terms, says a fix is coming, and gives a rough timeline.

> Ah okay I think I'm onto something. The fact that all those NaN blocks start right at the 10,000 indices (30k, 40k, 50k) and 10,000 being the size of the chunks used in the computation gave me a hint. I was able to get Claude to reproduce the bug on a toy example and I think I see what's going on. Let me work on a fix and get back to you later today / tomorrow. I think this will require releasing a keypoint-moseq patch. Thanks for bringing this to our attention!

**Sending them something to try:** one paragraph. An opener, the steps with the order in caps, one plain sentence on what the change does, and the conditional next step. That is the whole email: side effects, fallback values and what-ifs wait until they come up.

> Alright, let's try this. Take the code in the .py file in there and paste it into a cell and run it AFTER you've imported keypoint_moseq but BEFORE calling kpms.estimate_syllable_marginals. You will have to run this in a fresh notebook where nothing has been run yet. Basically, we're replacing a function with a fixed version so that when you run the estimate syllable marginals, it will call the fixed version. If this works, I'll make it a keypoint-moseq patch.

**It worked:** thanks with some enthusiasm, one plain sentence on what was actually broken, and what happens next.

> Amazing! Thanks for reporting this and being our guinea pig! This code that is broken is owned by a package we depend on rather than us directly, so I will file an issue with them to get this fixed.
>
> Best,
> John

## Issues on other projects

A bug report to another project's maintainers is first person and plain, in this order:

1. What we saw, and the one combination where it happens versus the ones where it doesn't.
2. The exact call and versions, and where it stops. Stack traces and code go in code blocks.
3. What we can't do, said plainly, with a short apology if it means no minimal repro.
4. One specific question for them, with a hedge on our read of their internals.
5. An offer to run whatever would help them, and a link to the failing log.

Write each in words specific to this report. dask/distributed#9382 is one he approved.

If it turns out to be our bug, the retraction goes in the same thread: own it in the first line, say in a sentence or two what it really was and how we fixed it, then close it with thanks.

> Sorry for the noise - this turned out to be our bug, not yours.
>
> (what it really was, and the fix)
>
> Closing this. Thanks for any time spent looking at it.
