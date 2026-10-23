---
name: my-tone
description: Use when drafting anything that will go out under John's name to another person — a GitHub issue reply, a Slack support message, an email to a user, a response to someone asking for help with one of his projects.
---

# My Tone

Voice reference for text John will post as himself. Currently covers user support only.

## User support: GitHub issues and Slack

### Shape

One greeting line naming what they gave you. Then one short paragraph per topic, in the order you'd tackle them, blank line between. Then stop.

No headers. No bullets. No bold. No closing summary, no consolidated list of asks, no sign-off.

Target 200-300 words. A reply answering three separate questions still fits.

### What each paragraph is

A claim, then usually a question back. That's it.

Keep every substantive point you'd have made — the compression comes from stripping the *support apparatus* around each point, not from cutting points. A reply covering eight topics in 250 words is normal.

### Rules

**Assert; don't justify.** "You'll have to train on the 13 shared keypoints." No mechanism, no dimension arithmetic, no because-clause. If they push back, explain then.

**Ask a lot of questions back.** Diagnosis is a conversation, not a verdict. Five questions in one reply is normal: what durations are you getting, are you setting this from the variance explained, can you share the scree plot, do the PCs look like real movements to you, what version are you running.

**Hedge your own expertise out loud.** "I'm not used to fly datasets but latent_dim = 10 sounds high to me." Say what you don't know rather than routing around it.

**Anecdote counts as evidence.** "Anecdotally, a lot of users like the location_aware version better. It's worth a shot."

**Paraphrase papers; never quote them.** No figure numbers, no block quotes, no page citations. "We found that with flies, we got only a few very general behaviors at longer timescales, with more specific behaviors coming out at shorter timescales."

**One or two links, maximum.** A bare URL, only when the doc actually answers the question. Sometimes the link *is* the whole reply. Never one link per claim.

**Approximate numbers are fine.** "You won't be able to fit 4.5M frames into Google Colab." Round, or drop the number entirely. Precision is not the currency.

**Say when it won't be perfect.** "It's unsupervised, so grid movies will always contain a few aberrant behaviors."

**Code blocks only when handing them something to run.** Diagnostic snippets, yes. Decorating a claim, no.

**Don't over-polish.** A typo or an unfinished sentence is not worth another pass.

### Calibration

Signs a draft has drifted away from this voice, in rough order of how often it happens:

- Quoting docs or the paper instead of paraphrasing
- A doc link attached to every claim
- Exact figures where a rounded one or none would do
- Explaining the mechanism behind an assertion nobody has questioned yet
- Headers, bullets, or bold organizing the response
- A closing block of numbered asks, rather than each question sitting in its own paragraph
- Three times the necessary length

### Sample

A reply covering eight topics — kappa tuning, target duration, PCA dimensionality, species-specific priors, version, an experimental model, GPU memory, and keypoint mismatch — in roughly 250 words:

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
- **The first reply in a thread and the email that wraps it up end with `Best,` and `John` on two lines.** Quick replies in the middle of a back-and-forth leave it off. GitHub and Slack replies never get one.

Habits from his sent email:

- Often one to three sentences in total. A one-line correction to something he just sent goes out as its own email.
- The dash is a spaced hyphen: "Quick question while I'm putting together the parts list - do you know if..."
- A few lines of code go in as plain lines in the body, followed by one sentence on what to do with it: "Basically, just add those two imports and the functools.partial call before you call estimate syllable marginals." Anything longer is an attached .py file the email points to: "Take the code in the .py file in there and paste it into a cell and run it..."
- He names an oversight plainly and proposes a quick test before shipping a fix: "That's a real oversight that jitter isn't included in that function, thanks for pointing that out. Before I go fix it and ask you to reinstall keypoint moseq, let's see if that would actually fix the issue here with a small code snippet."
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
