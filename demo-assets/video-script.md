# YouTube Demo Video Script

Target length: 90 to 110 seconds. Target word count: 220 to 270 words at 150 wpm.
Audience: Meta PyTorch OpenEnv Hackathon judges. Optimized for the rubric (Innovation 40, Storytelling 30, Reward Improvement 20, Pipeline 10).

Recording plan: screen recording with voice-over. No talking head needed. Cursor follows what the voice describes. Use OBS or Loom. Export 1080p.

## Title card (0:00 to 0:05)

On screen:
```
OpsTwin Recovery Arena
Enterprise incident response, as an OpenEnv environment
Theme 3.1 - Shashwat Rajan, Tanish Shitanshu
```

Voiceover:
> OpsTwin Recovery Arena. An OpenEnv environment for training LLM agents on real enterprise incident response.

## The hook (0:05 to 0:20)

On screen: split window, left side shows a terminal with a service dashboard (`services` and `alerts` from `REQUEST_INFO`), right side shows the ChatML system prompt.

Voiceover:
> Every benchmark of enterprise agents shows the same failure. They can call a single API. They cannot run an incident bridge. Hidden state, cross-team coordination, policy constraints. These are what agents fail on. OpsTwin makes those failure modes the whole game.

## The environment (0:20 to 0:45)

On screen: quick cuts through the repo.
- Show `server/desks.py` scrolled so five desk names are visible.
- Show `server/hidden_state.py` header with the six hidden variables.
- Show `server/scoring.py` with the six dimension names.

Voiceover:
> Five operational desks. Incident Command, SRE, Security, Support, Release. Each has its own command set and a filtered view. Hidden state: root cause, blast radius, stale approvals, actual flag values. The agent reveals them through inspection actions, and every reward is pure deterministic Python. No LLM judge.

## The scenario (0:45 to 1:10)

On screen: run `python inference.py` live against the `bad_release` scenario. Trim to the first 5 steps plus the DONE line.

Expected output (illustrative):
```
S1  SWITCH_DESK INCIDENT_COMMAND    +0.010
S2  ASSESS_BLAST_RADIUS             +0.020   hidden edge: checkout -> analytics
S3  SWITCH_DESK RELEASE             +0.010
S4  VERIFY_FLAG checkout_v2_ui      +0.020   flag is on, likely root cause
S5  FLIP_FLAG checkout_v2_ui off    +0.150   outage resolved
...
FINAL SCORE: 0.88
```

Voiceover:
> Bad Release scenario. Checkout is degraded after a deploy, but the deploy is not the root cause. It's a feature flag. If the agent rolls back without assessing the blast radius, it cascades into an analytics service. The correct play is Verify Flag, then Flip Flag. The model learns this.

## The numbers (1:10 to 1:30)

On screen: the comparison bar chart (`results/eval_curve.png`) full screen.

Voiceover:
> We fine-tuned Qwen 3 1.7B on 3,600 expert trajectories. 16 minutes on a single 5090. Average score went from 0.24 random to 0.74 trained. On the Security CVE scenario, from 0.20 to 0.94. That's the reward improvement the rubric asks for.

## Call to action (1:30 to 1:45)

On screen: three URL cards, one per line.
```
huggingface.co/spaces/Deltasthic/opstwin-recovery
huggingface.co/Deltasthic/opstwin-qwen3-1.7b-sft
github.com/Deltasthicc/opstwin-recovery
```

Voiceover:
> The environment is live on HuggingFace Spaces. The trained model is on the Hub. The Colab notebook reproduces the full pipeline. Point your OpenEnv client at the Space and start training.

## Outro (1:45 to 1:55)

On screen: OpsTwin title card again with contact info.

Voiceover:
> OpsTwin Recovery Arena. Theme 3.1, World Modeling. Thanks for watching.

---

## Things you still need to do

### Recording
1. Install OBS Studio (free) or Loom. OBS gives you fine-grained cropping.
2. Record at 1920x1080, 30 fps, mp4. Mic at 48 kHz mono.
3. Before recording, `python -m server.app` in one terminal, `python inference.py` in another. Dry-run once so the output is in the terminal scrollback.

### Editing
1. Cut silences over 400 ms with DaVinci Resolve (free) or CapCut. Keep it tight.
2. Add a soft background track at low volume. YouTube Audio Library or Pixabay Music. No copyrighted music.
3. Export mp4 H.264, 8 Mbps bitrate, audio AAC 192 kbps.

### Publishing
1. YouTube title: "OpsTwin Recovery Arena: teaching LLMs to run incident response" (under 70 chars).
2. YouTube description:
   ```
   OpsTwin Recovery Arena is an OpenEnv environment for enterprise incident response. Built for the Meta PyTorch OpenEnv Hackathon Grand Finale, April 25-26 2026, Bangalore.

   Space: https://huggingface.co/spaces/Deltasthic/opstwin-recovery
   Model: https://huggingface.co/Deltasthic/opstwin-qwen3-1.7b-sft
   Code: https://github.com/Deltasthicc/opstwin-recovery

   Authors: Shashwat Rajan, Tanish Shitanshu
   Track: Theme 3.1 World Modeling, Professional Tasks
   ```
3. Thumbnail: the OpsTwin title card with the average-score jump (0.24 -> 0.74) overlayed in large text.
4. Set visibility Public. Save the URL to `demo-assets/youtube-link.txt`.

### After upload
1. Paste the YouTube URL into the blog post (`demo-assets/blog-post.md`, top of the article as a hero embed).
2. Paste the URL into the README.md, under the "Try it yourself" section.
3. Update `TODO.md`: mark W3 done.

## What you do NOT need

- No talking head shot. A clean screen recording with voice-over is faster to produce and easier to edit.
- No slide deck. The repo and terminal are the slides.
- No motion graphics. Simple title cards are enough.

## If you run out of time

The blog post alone satisfies the rubric's "mini-blog OR video" requirement. Publish the blog first. Video can land anytime before submission close.
