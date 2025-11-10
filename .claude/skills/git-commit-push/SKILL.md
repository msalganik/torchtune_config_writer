---
name: git-commit-push
description: This skill should be used when the user wants to commit their work to git and push to GitHub. It guides through reviewing changes, crafting meaningful commit messages following project conventions (including Conventional Commits when detected), creating commits with security checks, and pushing to remote repositories.
---

# Git Commit and Push

## Overview

This skill provides a structured workflow for committing changes to git and pushing to GitHub. It ensures changes are reviewed, commit messages are meaningful and follow conventions, secrets are not committed, and commits are properly pushed to remote repositories. The skill adapts to project conventions, supporting both Conventional Commits and custom formats.

## When to Use This Skill

Use this skill when the user:
- Explicitly requests to "commit my work" or "push to GitHub"
- Says they want to save/commit/push their changes
- Asks to create a commit or push code
- Wants to share their work on GitHub

## Workflow

### Step 1: Review Changes

Before committing, review what has changed:

1. **Check git status** to see modified, new, and deleted files:
   ```bash
   git status
   ```

2. **Analyze diff systematically**:
   ```bash
   git diff --stat  # Overview of changes
   git diff         # Detailed line-by-line changes
   ```

3. **Categorize changes** to inform commit message:
   - **New features**: New files, new functions, new capabilities
   - **Bug fixes**: Modified logic, error handling improvements
   - **Refactoring**: Structure changes with no behavior change
   - **Documentation**: *.md files, code comments, docstrings
   - **Tests**: Test files, test additions/modifications
   - **Configuration**: Build files, dependencies, settings
   - **Styling**: Formatting, whitespace, code style only

4. **Check recent commits** to understand commit message style:
   ```bash
   git log --oneline -20
   ```

5. **Present summary** to the user:
   - List all modified, new, and deleted files
   - Highlight key changes by category
   - Note total lines added/removed
   - Flag any unusual patterns (large files, many deletions, etc.)

### Step 2: Detect Project Convention

Determine if project uses Conventional Commits or custom format:

1. **Check for Conventional Commits pattern**:
   ```bash
   # Look for type(scope): format in recent commits
   git log --oneline -20 | grep -E "^[a-f0-9]+ (feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?:"
   ```

2. **Analyze result**:
   - If ≥50% of commits match pattern → Project uses Conventional Commits
   - Otherwise → Project uses custom format

3. **Note the convention** for commit message crafting

### Step 3: Craft Commit Message

Create a meaningful commit message following detected convention:

#### If Conventional Commits Detected

Use format: `type(scope): description`

**Common types**:
- `feat`: New feature or capability
- `fix`: Bug fix
- `docs`: Documentation changes only
- `style`: Code style/formatting (no logic change)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or modifying tests
- `chore`: Maintenance, dependencies, build
- `perf`: Performance improvements
- `ci`: CI/CD pipeline changes
- `build`: Build system or external dependencies

**Optional scope**: Component or module affected (e.g., `auth`, `api`, `ui`)

**Format**:
```
type(scope): imperative summary (max 50 chars)

Optional body explaining the change in detail.
Explain WHY, not WHAT (the diff shows what).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Example**:
```
feat(evaluation): add Inspect AI integration to experiment framework

Implements Phase 4 evaluation support with:
- User-written evaluation tasks workflow
- Adapter-only checkpointing (200x storage reduction)
- Complete working examples and documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

#### If Custom Format Detected

Match the project's existing style:

1. **Summarize changes** in 1-2 sentences (imperative, present tense)
2. **Focus on "why"** rather than "what" (diff shows what changed)
3. **Follow observed conventions** (capitalization, punctuation, length)
4. **Keep concise** but informative

**Format**:
```
Summary line (imperative mood, present tense)

Optional detailed explanation of the changes.
Focus on motivation and context.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Example**:
```
Add evaluation configuration to experiment definition framework

Implements Inspect AI integration, adapter-only checkpointing support,
and complete working examples for Phase 4 implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Step 4: Check for Secrets

Before staging, scan for sensitive information:

1. **Check unstaged changes for secret patterns**:
   ```bash
   git diff | grep -E "(api[_-]?key|api[_-]?secret|password|token|secret[_-]?key|private[_-]?key|aws[_-]?access)" -i
   ```

2. **Check for specific patterns**:
   - AWS Access Keys: `AKIA[0-9A-Z]{16}`
   - Generic API keys: `[a-zA-Z0-9_-]{32,}`
   - Private keys: `-----BEGIN.*PRIVATE KEY-----`
   - GitHub tokens: `ghp_`, `gho_`, `ghs_`, `ghr_`
   - Bearer tokens: `Bearer [a-zA-Z0-9._-]+`
   - Passwords in configs: `password\s*[:=]\s*["']?[^"'\s]+`

3. **Check for sensitive files**:
   ```bash
   git status --short | grep -E "\.env|credentials|secrets|\.pem|\.key|\.p12"
   ```

4. **If secrets detected**:
   - ❌ **STOP immediately**
   - Show user the matched patterns (without revealing full values)
   - Explain the risk
   - Ask user to:
     - Remove secrets from code
     - Use environment variables or secret management
     - Add file to `.gitignore` if appropriate
   - **Do not proceed** until secrets are removed

5. **If no secrets detected**:
   - ✅ Proceed to staging

### Step 5: Stage and Commit Changes

1. **Stage relevant files**:
   ```bash
   git add <file1> <file2> ...
   ```

   **Staging guidelines**:
   - Stage files related to this logical change
   - Do NOT stage unrelated changes (save for separate commit)
   - If uncertain, ask user which files to include
   - Show user which files will be staged

2. **Verify staged changes**:
   ```bash
   git diff --staged --stat
   ```

3. **Create commit** with formatted message:
   ```bash
   git commit -m "$(cat <<'EOF'
   <commit message here>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

4. **Verify commit** was created:
   ```bash
   git log -1 --stat
   ```

### Step 6: Push to Remote

1. **Check remote status**:
   ```bash
   git status  # Check if branch tracks remote
   ```

2. **Push to GitHub**:
   ```bash
   # If branch already tracks remote
   git push

   # If new branch (first push)
   git push -u origin <branch-name>
   ```

3. **Confirm success**:
   - Report push result to user
   - Show number of commits pushed
   - Provide remote URL if available
   - Note any warnings or issues

## Important Guidelines

### Safety Rules

- **Never push force** to main/master without explicit user confirmation
- **Never skip hooks** (--no-verify) unless explicitly requested by user
- **Never commit secrets** - stop and warn user if detected
- **Check authorship** before amending commits (don't amend others' work)
- **Check for large files** - warn if files > 100KB (may need Git LFS)
- **Always verify** what's being committed before executing

### Best Practices

1. **Review before committing** - show user what will be committed
2. **Meaningful messages** - explain why changes were made, not just what
3. **Atomic commits** - each commit should be a single logical change
4. **Follow conventions** - detect and match project's commit style
5. **Add co-author attribution** - include Claude Code footer
6. **Keep commits focused** - don't mix refactoring with features

### Commit Scope Guidelines

**Good commit scope**:
- ✅ One logical change per commit
- ✅ Related files together (e.g., code + corresponding test)
- ✅ Can be described in one sentence
- ✅ Typically < 500 lines (unless it's a rename/refactor)
- ✅ All changes serve the same purpose

**Signs to split commits**:
- ❌ Using "and" more than once in commit message
- ❌ Changes span multiple unrelated features
- ❌ Mix of bug fixes and new features
- ❌ Experimental changes alongside production code
- ❌ "WIP", "misc fixes", or "various updates" messages

**When to split**:
```bash
# If you have unrelated changes, stage selectively:
git add -p  # Interactively choose hunks to stage

# Or stage specific files:
git add file1.py file1_test.py  # First commit: feature A
git commit -m "feat: add feature A"

git add file2.py file2_test.py  # Second commit: feature B
git commit -m "feat: add feature B"
```

### Error Handling

**If commit fails**:
- Check for pre-commit hooks modifying files
- Review error message carefully
- May need to stage hook-modified files and amend
- Only amend if: (1) user requested it OR (2) fixing pre-commit hook changes AND it's safe (check authorship, not pushed)

**If secrets detected**:
- Stop immediately, do not commit
- Explain which patterns matched
- Guide user to remove secrets
- Suggest using environment variables or secret management
- Consider adding `.gitignore` entries

**If push fails**:
- Check remote branch status (might need pull first)
- Check for merge conflicts
- Verify remote URL is correct
- Check authentication/permissions
- Report error to user with suggested fixes

**If large files detected**:
- Warn user about file sizes
- Suggest Git LFS for files > 100KB
- Ask if files should be committed or added to `.gitignore`

## Example Interaction

### Example 1: Conventional Commits Project

**User**: "I'd like to commit my work and push to GitHub"

**Assistant**:
1. Runs `git status` and `git diff --stat`
2. Analyzes: "You've added 2 files (1,200 lines) and modified 1 file (69 lines)"
3. Categorizes: "Changes are: new feature (eval config), documentation (README)"
4. Checks convention: Detects Conventional Commits (15/20 recent commits match)
5. Drafts: `feat(evaluation): add Inspect AI integration to experiment framework`
6. Scans for secrets: None detected ✓
7. Shows proposed commit to user
8. Stages: `git add SPEC.md appendices/F_experiment_definition.md`
9. Commits with Conventional Commits format
10. Pushes: `git push`
11. Reports: "✓ 1 commit pushed to GitHub successfully"

### Example 2: Custom Format Project, Secrets Detected

**User**: "commit my changes"

**Assistant**:
1. Reviews changes: Modified config.py (+10 lines)
2. Scans for secrets: ❌ **FOUND** pattern matching API key
3. **STOPS**: "⚠️ Security Warning: Detected potential API key in config.py"
4. Shows matched line: `api_key = "sk_live_..."`
5. Advises: "Please use environment variables instead: `api_key = os.getenv('API_KEY')`"
6. Asks: "Would you like to remove the secret before committing?"
7. **Does not proceed** until user confirms secrets are removed

## Notes

- This skill follows the git safety protocol from the Bash tool documentation
- Commit message format includes Claude Code attribution as per project conventions
- Always run commands sequentially (staging, committing, pushing) not in parallel
- Adapts to project conventions rather than enforcing a single standard
- Security checks are non-negotiable - will not commit secrets
- If uncertain about any operation, ask the user for confirmation

## References

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Git Commit Best Practices](https://cbea.ms/git-commit/)
- [Pre-commit Framework](https://pre-commit.com/)
- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
