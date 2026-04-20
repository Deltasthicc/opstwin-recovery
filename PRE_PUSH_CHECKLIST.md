# Pre-Push Checklist

Before running `git push`, walk this list top to bottom. Any item that fails stops the push.

## 1. Secrets scan

No tokens or keys in committed files. Hugging Face tokens start with `hf_`.

```powershell
# Windows (PowerShell)
git diff --cached | Select-String -Pattern "hf_[A-Za-z0-9]{20,}", "sk-[A-Za-z0-9]{20,}"
```

```bash
# macOS/Linux
git diff --cached | grep -E "hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"
```

Both should produce **no output**. If anything prints, remove the secret, regenerate it on the provider side (you can't un-leak a published token), and restart this checklist.

Also confirm `.env` is not staged:

```powershell
git status | Select-String ".env"
```

Only `.env.example` should appear. If `.env` appears, your `.gitignore` didn't load — fix and retry.

## 2. All tests pass

```powershell
python -m pytest tests/test_environment.py -q
```

Expected: `256 passed`. If any fail, either the code is broken or the tests are stale. Do not push with red tests.

## 3. Local server starts cleanly

```powershell
python -m server.app
```

Another PowerShell window:

```powershell
curl http://localhost:8000/health
```

Should print `{"status":"healthy"}`. Ctrl-C the server window when done.

## 4. Docker build succeeds

```powershell
docker build -t opstwin-recovery .
```

Must end with `Successfully tagged opstwin-recovery:latest`. If this fails on your machine it will fail on judges' machines too.

## 5. README is up to date

Open `README.md` and check:
- The "How it's built" tree matches your actual file layout
- Training command example uses your actual default model
- All file paths in the tree exist

## 6. No large files accidentally committed

```powershell
git ls-files | ForEach-Object { $size = (Get-Item $_).Length; if ($size -gt 1MB) { "$_ : $($size/1MB) MB" } }
```

Anything over 1 MB should be deliberate (training checkpoints are blocked by `.gitignore` and should not appear here). Trajectory JSONL files are fine (small, important evidence). If a chart PNG is > 1 MB, regenerate at lower DPI.

## 7. Remove runtime junk

```powershell
Remove-Item -Recurse -Force __pycache__, server\__pycache__, tests\__pycache__, baselines\__pycache__, .pytest_cache, *.egg-info -ErrorAction SilentlyContinue
```

## 8. Expert trajectories up to date (optional but recommended)

```powershell
python baselines/expert_solver.py
```

Re-generates the gold traces used by `tests/test_environment.py::TestIntegrationOptimal`. Commit them if they're changed — they're evidence for judges.

## 9. Final smoke check

```powershell
python tests/smoke_all_scenarios.py
```

Expected: `bad_release`, `security_cve`, and `data_pipeline_regression` all print `final=0.9900 resolved=N/N`. If any score is below 0.95, the expected-trajectory math has drifted and the trajectories need attention.

## 10. Commit message

Follow conventional-commit style:

```
feat: add procedural scenario generator for bad_release family
fix: scoring cap for zero-progress episodes
docs: update README training instructions
test: expand environment test suite to 256 cases
```

Keep the first line under 72 characters. Include a body if the change needs context.

## 11. Push

```powershell
git push origin main
```

If pushing to a new remote for the first time:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/opstwin-recovery.git
git branch -M main
git push -u origin main
```

---

## If you realise you pushed a secret

1. **Immediately** revoke the token at https://huggingface.co/settings/tokens (or the equivalent page for your provider). This is the only real fix — the history is public the moment you push.
2. Regenerate a new token.
3. Update your local `.env`.
4. Consider using `git filter-repo` to rewrite history (removes the leak from past commits), then force-push. This is disruptive if anyone has cloned already.
5. Never assume "I'll just delete the commit." GitHub caches deleted commits by SHA for at least 90 days and any bot watching the events feed saw it within seconds.
