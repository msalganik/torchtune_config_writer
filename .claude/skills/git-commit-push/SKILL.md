---
name: git-commit-push
description: This skill should be used when the user wants to commit their work to git and push to GitHub. It guides through reviewing changes, crafting meaningful commit messages following project conventions, creating commits, and pushing to remote repositories.
---

# Git Commit and Push

## Overview

This skill provides a structured workflow for committing changes to git and pushing to GitHub. It ensures changes are reviewed, commit messages are meaningful and follow conventions, and commits are properly pushed to remote repositories.

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

2. **Review diffs** for key files to understand the changes:
   ```bash
   git diff <important-files>
   ```

3. **Check recent commits** to understand commit message style:
   ```bash
   git log --oneline -5
   ```

4. **Present summary** to the user:
   - List all modified files
   - Highlight key changes
   - Note any untracked files

### Step 2: Craft Commit Message

Create a meaningful commit message that:

1. **Summarizes the changes** in 1-2 sentences
2. **Focuses on "why"** rather than "what" (the diff shows the "what")
3. **Follows project conventions** based on recent commit messages
4. **Is concise** but informative

**Standard format**:
```
<summary line - present tense, imperative mood>

<optional detailed explanation>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Example commit messages**:
```
Add evaluation configuration to experiment definition framework

Includes Inspect AI integration, adapter-only checkpointing support,
and complete working examples for Phase 4 implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Step 3: Stage and Commit Changes

1. **Stage relevant files**:
   ```bash
   git add <file1> <file2> ...
   ```

   - Stage new files and modifications
   - Do NOT stage sensitive files (.env, credentials, secrets)
   - If uncertain, ask user which files to include

2. **Create commit** with formatted message:
   ```bash
   git commit -m "$(cat <<'EOF'
   <commit message here>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

3. **Verify commit** was created:
   ```bash
   git log -1
   ```

### Step 4: Push to Remote

1. **Check remote status**:
   ```bash
   git status  # Check if branch tracks remote
   ```

2. **Push to GitHub**:
   ```bash
   # If branch already tracks remote
   git push

   # If new branch
   git push -u origin <branch-name>
   ```

3. **Confirm success**:
   - Report push result to user
   - Provide remote URL if available
   - Note any warnings or issues

## Important Guidelines

### Safety Rules

- **Never push force** to main/master without explicit user confirmation
- **Never skip hooks** (--no-verify) unless explicitly requested
- **Never commit secrets** - warn user if sensitive files detected
- **Check authorship** before amending commits (don't amend others' commits)

### Best Practices

1. **Review before committing** - show user what will be committed
2. **Meaningful messages** - explain why changes were made
3. **Atomic commits** - each commit should be a logical unit
4. **Follow conventions** - match existing commit message style
5. **Add co-author attribution** - include Claude Code footer

### Error Handling

If commit fails:
- Check for pre-commit hooks modifying files
- Review error message carefully
- May need to stage hook-modified files and amend
- Only amend if: (1) user requested it OR (2) fixing pre-commit hook changes AND it's safe (check authorship, not pushed)

If push fails:
- Check remote branch status (might need pull first)
- Verify remote URL is correct
- Check authentication/permissions
- Report error to user with suggested fixes

## Example Interaction

**User**: "I'd like to commit my work and push to GitHub"

**Assistant**:
1. Runs `git status` and `git diff` to review changes
2. Summarizes: "You've updated 3 files: SPEC.md, appendices/F_experiment_definition.md, and created 5 new example files"
3. Drafts commit message: "Add evaluation configuration to experiment definition framework"
4. Shows proposed commit message to user
5. Stages files: `git add SPEC.md appendices/F_experiment_definition.md example_configs/...`
6. Creates commit with proper formatting
7. Pushes to GitHub: `git push`
8. Reports: "✓ Changes committed and pushed to GitHub successfully"

## Notes

- This skill follows the git safety protocol from the Bash tool documentation
- Commit message format includes Claude Code attribution as per project conventions
- Always run commands sequentially (staging, committing, pushing) not in parallel
- If uncertain about any operation, ask the user for confirmation
