# Wireframe Technical Specification

Generate a comprehensive wireframe document from a technical specification with visual diagrams, concrete examples, and before/after comparisons.

## What This Skill Does

Takes a technical specification and creates a user-friendly wireframe document that shows:
- Visual architecture diagrams (ASCII art)
- Concrete code examples with actual output
- Before/after comparisons
- Step-by-step workflows with visual flow
- File contents and directory structures
- Real-world scenarios and use cases

## Instructions

You are helping create a wireframe document to demonstrate a technical specification to colleagues.

### Step 1: Understand the Specification

First, read the specification file:
- Identify the core problem being solved
- Note the target use case and primary users
- List key features and capabilities
- Understand the system architecture
- Identify what makes this tool unique/valuable

### Step 2: Identify Source Files

Ask the user:
1. What are the INPUT files/configs that the tool works with?
2. Where do these source files come from? (Package installation, user files, APIs, etc.)
3. What format are they in? (YAML, JSON, Python dicts, etc.)

Then create a section explaining:
- Where source files come from
- How to access/list available sources
- What's inside a typical source file (show actual example)
- Visual diagram of source → tool → output flow

### Step 3: Create System Architecture Visuals

Generate ASCII art diagrams showing:

**High-Level Data Flow:**
```
┌─────────────────────┐
│  User Input         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Tool Processing    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Generated Output   │
└─────────────────────┘
```

**Key Algorithm/Process:**
- Show visual representation of core logic (e.g., merge semantics, transformation rules)
- Use side-by-side comparisons
- Show input → transformation → output

### Step 4: Create Problem/Solution Comparison

Start the wireframe with a clear problem statement:

**Without This Tool (Current Pain):**
- Show manual workflow
- Highlight pain points with ❌
- Include time estimates
- Show error scenarios

**With This Tool (Solution):**
- Show automated workflow
- Highlight benefits with ✓
- Compare time/effort
- Emphasize safety/validation

### Step 5: Generate Concrete Examples

For each major use case, create:

#### Example Structure:
1. **Scenario** - Clear description of what user wants to accomplish
2. **Visual Workflow** - ASCII diagram showing step-by-step process
3. **Your Code** - Actual Python/code user would write
4. **Console Output** - What user sees (with ✓ and timing info)
5. **Files Generated** - Show actual file contents (YAML, JSON, etc.)
6. **Why This Matters** - Before/after comparison or key insight

### Step 6: Add Visual Diagrams Throughout

For complex workflows, create visual diagrams showing:
- Data flow between components
- State changes over time
- Inheritance/lineage trees
- Parallel execution (e.g., SLURM job arrays)
- Timeline diagrams (what happens when)

**Diagram Style:**
- Use box drawing characters: ┌─┐│└┘├┤┬┴┼
- Use arrows: → ← ↑ ↓ ▼ ▲
- Use symbols: ✓ ✗ ⚠️ ⭐ ◄──
- Keep width under 70 characters for readability
- Add labels and annotations

### Step 7: Show File System Results

After examples, show:
```
directory_structure/
├── user_script.py       # What user writes
├── generated/
│   ├── output1.yaml     # Generated files
│   ├── output1.meta.yaml
│   └── output2.yaml
└── logs/
    └── execution.log
```

Include actual file contents with annotations:
```yaml
# config.yaml
key: value  # ◄── Customized
other: inherited  # ◄── From base
```

### Step 8: Create Comparison Tables

Add tables comparing:
- Manual vs. automated approach
- Time/effort metrics
- Error rates
- Scalability
- Feature coverage

Use clear formatting:
```
┌──────────────┬──────────┬──────────┐
│ Metric       │ Manual   │ Tool     │
├──────────────┼──────────┼──────────┤
│ Time         │ 2-3 hrs  │ 3 sec    │
│ Errors       │ High ⚠️  │ Zero ✓   │
└──────────────┴──────────┴──────────┘
```

### Step 9: End-to-End Workflow

Create a complete workflow diagram showing:
1. Initial setup
2. Generation phase (with timing)
3. Validation phase
4. Execution/deployment phase
5. Results phase
6. Iteration phase (optional)

Include timeline annotations (0s, 3s, 6 hours, etc.)

### Step 10: Key Takeaways Section

Summarize with:
- What makes this tool valuable (numbered list)
- Who benefits (user personas)
- Comparison table (before/after)
- Success metrics

## Output Format

Create a markdown file named `WIREFRAME.md` with:

```markdown
# [Tool Name] - Wireframe & Examples

**Purpose**: Demonstrate how the tool works with concrete examples
**Audience**: [Target audience]
**Date**: [Current date]

## Table of Contents

1. [Where Do Source Files Come From?](#where-do-source-files-come-from)
2. [System Architecture (Visual)](#system-architecture-visual)
3. [The Problem We're Solving](#the-problem-were-solving)
4. [Example 1: Simple Use Case](#example-1-simple-use-case)
5. [Example 2: Power Feature](#example-2-power-feature)
6. [Example 3: Advanced Scenario](#example-3-advanced-scenario)
7. [Complete Workflow](#complete-workflow)
8. [What Files Get Generated](#what-files-get-generated)
9. [Key Takeaways](#key-takeaways)

[Rest of wireframe...]
```

## Best Practices

1. **Be Concrete**: Use actual code, not pseudocode. Show real file paths, real commands.

2. **Show, Don't Tell**: Visual diagrams before text explanations. Code before theory.

3. **Include Timing**: "3 seconds", "6 hours", "2-3 hours" - concrete numbers matter.

4. **Real File Contents**: Show complete, valid file contents that could actually be used.

5. **Annotations**: Use inline comments (# ◄──) to highlight changes/important parts.

6. **Error Scenarios**: Show what happens when things go wrong, not just success cases.

7. **Comparison Tables**: Always compare old way vs. new way with metrics.

8. **Progressive Complexity**: Start simple, build to complex scenarios.

9. **Visual Consistency**: Use same ASCII art style throughout document.

10. **Emphasize Value**: Every example should make clear "why this matters".

## Example Prompt Flow

1. Read the spec file
2. Ask user: "What are the input files this tool works with?"
3. Ask user: "What specific scenarios should I emphasize in the wireframe?"
4. Generate wireframe with:
   - Source file explanation section
   - Visual architecture diagrams
   - 3-5 concrete examples with full code and output
   - Complete end-to-end workflow
   - Comparison tables and key takeaways

## Success Criteria

The wireframe should:
- ✓ Allow someone unfamiliar with the spec to understand the tool in 5-10 minutes
- ✓ Include copy-paste-able code examples
- ✓ Show actual file contents and directory structures
- ✓ Visualize complex workflows with ASCII diagrams
- ✓ Make the value proposition immediately clear
- ✓ Include before/after comparisons with concrete metrics
- ✓ Cover simple → advanced use cases progressively
- ✓ Show where source files come from and how to access them

## Notes

- The wireframe complements the spec, doesn't replace it
- Focus on user's perspective (what they write, what they see)
- Every visual should have a purpose
- Keep code examples realistic and runnable
- Show the "why" not just the "how"

---

## Review Mode

Use this section when reviewing an existing wireframe to improve it.

### How to Use Review Mode

Ask the user: "Would you like me to review and improve the existing wireframe?"

If yes, follow this review checklist:

### Review Checklist

#### 1. **Source File Clarity** ✓/✗
- [ ] Does it explain where input files come from?
- [ ] Shows how to list/access available sources?
- [ ] Includes actual example of source file contents?
- [ ] Has visual diagram of source → tool → output flow?

**If missing:** Add "Where Do Source Files Come From?" section at the beginning.

---

#### 2. **Visual Diagrams** ✓/✗
- [ ] System architecture diagram present?
- [ ] Data flow visualized with ASCII art?
- [ ] Each major example has step-by-step workflow diagram?
- [ ] Key algorithms/processes shown visually?
- [ ] Complex concepts broken down with diagrams?

**If missing:** Add ASCII diagrams showing:
- High-level architecture
- Step-by-step workflows for each example
- Data transformation processes
- Timeline diagrams for long-running operations

---

#### 3. **Problem/Solution Framing** ✓/✗
- [ ] Clear "Without This Tool" section showing pain?
- [ ] Concrete "With This Tool" showing benefits?
- [ ] Pain points marked with ❌?
- [ ] Benefits marked with ✓?
- [ ] Includes time/effort comparisons?

**If missing:** Add comparison section at the beginning showing:
```
Without This Tool:
❌ Manual process takes 2-3 hours
❌ Error-prone
❌ Hard to scale

With This Tool:
✓ Automated process takes 3 seconds
✓ Validated before execution
✓ Scales to 100+ items
```

---

#### 4. **Concrete Examples** ✓/✗
- [ ] At least 3-5 examples covering different use cases?
- [ ] Each example has real, runnable code?
- [ ] Console output shown (not just code)?
- [ ] Generated file contents displayed?
- [ ] Examples progress from simple → complex?

**If missing:** For each example, ensure it has:
1. Clear scenario description
2. Visual workflow diagram
3. Actual code user would write
4. Console output with timing
5. Generated file contents
6. "Why This Matters" explanation

---

#### 5. **File Contents & Structure** ✓/✗
- [ ] Shows actual generated file contents (YAML, JSON, etc.)?
- [ ] Uses inline annotations (# ◄──) to highlight changes?
- [ ] Directory structure shown after running examples?
- [ ] File sizes or complexity indicators included?

**If missing:** Add sections showing:
```yaml
# Actual file content
key: value  # ◄── Customized by user
other: default  # ◄── Inherited from base
```

---

#### 6. **Comparison Tables** ✓/✗
- [ ] Metrics comparing old vs. new approach?
- [ ] Concrete numbers (time, effort, error rate)?
- [ ] Clear formatting with borders?
- [ ] Symbols (✓, ✗, ⚠️) used effectively?

**If missing:** Add comparison tables like:
```
┌──────────────┬──────────┬──────────┐
│ Metric       │ Before   │ After    │
├──────────────┼──────────┼──────────┤
│ Time         │ 2 hrs    │ 3 sec    │
│ Errors       │ High ⚠️  │ Zero ✓   │
└──────────────┴──────────┴──────────┘
```

---

#### 7. **End-to-End Workflow** ✓/✗
- [ ] Complete workflow from start to finish?
- [ ] Includes all phases (setup, generation, validation, execution, results)?
- [ ] Timeline annotations (0s, 3s, 6 hours)?
- [ ] Shows iteration/feedback loop?
- [ ] Visual diagram of complete workflow?

**If missing:** Add comprehensive workflow diagram showing all steps with timing.

---

#### 8. **Timing & Performance** ✓/✗
- [ ] Concrete time estimates ("3 seconds", "6 hours")?
- [ ] Performance comparisons (manual vs. automated)?
- [ ] Scalability examples (10 items vs. 100 items)?

**If missing:** Add specific timing throughout:
- "Generated 16 configs in 3.2 seconds"
- "Validation completes in < 1 second per config"
- "Saves 2-3 hours compared to manual editing"

---

#### 9. **Value Proposition** ✓/✗
- [ ] Clear explanation of what makes tool unique?
- [ ] "Why This Matters" sections for complex examples?
- [ ] Who benefits from using this tool?
- [ ] Key takeaways section at end?

**If missing:** Add explicit value statements:
- Key differentiators
- Target user personas
- Success metrics
- ROI calculations (time saved, errors prevented)

---

#### 10. **Readability & Flow** ✓/✗
- [ ] Table of contents with all sections?
- [ ] Progressive complexity (simple → advanced)?
- [ ] Consistent formatting throughout?
- [ ] Section headers clear and descriptive?
- [ ] Code blocks properly formatted?
- [ ] Visual elements enhance understanding (not distract)?

**If missing:** Reorganize sections for better flow:
1. Source files explanation
2. Architecture overview
3. Problem/solution
4. Simple example
5. Power features
6. Advanced scenarios
7. Complete workflow
8. Key takeaways

---

### Review Output Format

After reviewing, provide a report in this format:

```markdown
## Wireframe Review Report

### Summary
[1-2 sentence overall assessment]

### Strengths ✓
- [What's working well]
- [Strong examples or visuals]
- [Clear explanations]

### Areas for Improvement ✗
1. **[Category]**: [Specific issue]
   - Suggestion: [How to fix]
   - Priority: High/Medium/Low

2. **[Category]**: [Specific issue]
   - Suggestion: [How to fix]
   - Priority: High/Medium/Low

### Missing Elements
- [ ] Source file explanation
- [ ] Visual diagrams in Example X
- [ ] Comparison table
- [ ] End-to-end workflow diagram
- [ ] Performance metrics

### Recommendations

**High Priority (Do First):**
1. [Most important improvement]
2. [Second most important]

**Medium Priority:**
1. [Nice to have improvements]

**Low Priority:**
1. [Polish and refinements]

### Proposed Changes

[If user agrees, I will make these specific changes:]
- Add section X with content Y
- Enhance example Z with diagram
- Create comparison table for feature W
```

---

### Review Mode Usage

**User invokes review mode:**
```
User: "Can you review the wireframe and suggest improvements?"

Assistant response:
1. Read the existing wireframe file
2. Apply the 10-point checklist
3. Generate review report with specific findings
4. Ask: "Would you like me to implement these improvements?"
5. If yes, make the changes and explain what was added/modified
```

### Interactive Review Process

**Step 1: Initial Assessment**
- Read wireframe file
- Check against 10 criteria
- Note what's strong vs. what's missing

**Step 2: Generate Report**
- Provide structured review report (use template above)
- Prioritize recommendations (High/Medium/Low)
- Be specific about what to add/change

**Step 3: Get User Approval**
Ask: "I found [X] areas for improvement. Would you like me to:"
- A) Make all high-priority improvements
- B) Let you choose which improvements to make
- C) Just provide the report without making changes

**Step 4: Implement Improvements**
- Add missing sections
- Enhance existing examples with visuals
- Insert comparison tables and timing info
- Improve formatting and flow

**Step 5: Summary**
After changes, provide:
```markdown
## Changes Made

### Added:
- [New section or diagram]
- [Enhanced example]

### Enhanced:
- [Existing section improved]
- [Added visuals to example]

### Fixed:
- [Formatting issues]
- [Missing metrics]

### Result:
The wireframe now includes [X] visual diagrams, [Y] concrete examples,
and [Z] comparison tables. It covers the full workflow from source files
to results with timing information throughout.
```

---

### Quick Review Commands

For efficient reviews, recognize these user requests:

- **"Review the wireframe"** → Full 10-point review with report
- **"Check if visuals are missing"** → Focus on criteria #2
- **"Is the value proposition clear?"** → Focus on criteria #9
- **"Add more examples"** → Focus on criteria #4
- **"Make it more visual"** → Focus on criteria #2, #7
- **"Show actual file contents"** → Focus on criteria #5
- **"Add before/after comparisons"** → Focus on criteria #3, #6

---

### Review Mode Example

```markdown
User: "Can you review WIREFRAME.md?"

Assistant: "I'll review WIREFRAME.md using the 10-point checklist."

[Reads file and applies checklist]

## Wireframe Review Report

### Summary
The wireframe is comprehensive with excellent visual diagrams and concrete examples.
It covers 5 major use cases with real code and timing metrics. A few enhancements
would improve clarity around file contents and add more comparison tables.

### Strengths ✓
- Excellent ASCII diagrams throughout (architecture, workflows, lineage)
- Comprehensive source file explanation with visual diagram
- Strong problem/solution framing with clear pain points
- 5 concrete examples with real code and console output
- Complete end-to-end SLURM workflow with timeline
- Actual YAML file contents shown with annotations
- Multiple comparison tables with concrete metrics
- Clear value proposition and key takeaways

### Areas for Improvement ✗
1. **File Contents**: Example 2 and 3 could show more complete generated files
   - Suggestion: Add full YAML content for at least 2 configs from the sweep
   - Priority: Medium

2. **Performance Metrics**: Add scalability examples
   - Suggestion: Show "10 configs vs. 100 configs" comparison
   - Priority: Low

### Missing Elements
- [ ] None - all major criteria met!

### Recommendations

**High Priority (Do First):**
- None - wireframe is complete

**Medium Priority:**
1. Add complete YAML contents for 2 configs in Example 2 (parameter sweep)
2. Add scalability comparison table (10 vs 100 vs 1000 configs)

**Low Priority:**
1. Add error scenario example (what happens when validation fails)
2. Add troubleshooting section

### Proposed Changes

If you'd like, I can:
- Add full YAML file contents to Example 2 (show 2 complete configs from sweep)
- Add a scalability comparison table in the Key Takeaways section
- Add an error handling example

Would you like me to make these medium-priority improvements?
```

This example shows what a complete review cycle looks like with specific, actionable feedback.
