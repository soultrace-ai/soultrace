---
title: "How Soultrace Works: A Deep-Dive for Nerds"
authors: ["Francesco Zuppichini", "Francesco Cicala"]
createdAt: "2025-01-03"
description: "The math and code behind Soultrace's Bayesian adaptive personality test. Hypothesis testing, entropy-based question selection, and why it actually works."
tags:
  [
    "bayesian-inference",
    "information-theory",
    "active-learning",
    "machine-learning",
    "personality-test",
    "soultrace",
  ]
---

# How Soultrace Works: A Technical Deep-Dive

Most personality tests are glorified surveys. Fixed questions, fixed order, dump answers into a spreadsheet, calculate percentages. Boring. Statistically lazy.

Soultrace uses **Bayesian inference** with **adaptive question selection**. Each question is chosen in real-time to test our current hypothesis about you. This post explains the actual math and code behind it.

## The Problem with Fixed Questionnaires

Traditional personality tests ask everyone the same 50-100 questions. This is inefficient for two reasons:

1. **Redundancy**: If your first 5 answers strongly indicate you're analytical, asking 10 more "analytical vs creative" questions wastes everyone's time
2. **Missed precision**: Fixed tests can't probe ambiguous areas - they treat confident classifications the same as uncertain ones

What we want: a test that **adapts** based on what it already knows, asking questions that **test its current hypothesis**.

## The Setup

We classify users into one of five archetypes (we call them colors: White, Blue, Black, Red, Green). Each archetype represents a fundamental drive:

| Color | Core Drive |
|-------|------------|
| White | Order, fairness, structure |
| Blue | Understanding, mastery, precision |
| Black | Agency, power, achievement |
| Red | Intensity, expression, action |
| Green | Connection, growth, harmony |

We maintain a probability distribution over these five archetypes, updated after each answer.

## Step 1: Bayesian Updates

We start with a uniform prior - equal probability for all archetypes:

```typescript
let distribution = {
  white: 0.2,
  blue: 0.2,
  black: 0.2,
  red: 0.2,
  green: 0.2
}
```

Each question has pre-computed **likelihood tables** - the probability that someone of each archetype would give each score on a 7-point Likert scale (1 = strongly disagree, 7 = strongly agree):

```typescript
const question = {
  text: "I prefer having a detailed plan before starting a project",
  scoreProbabilities: {
    white: { 1: 0.03, 2: 0.05, 3: 0.10, 4: 0.18, 5: 0.24, 6: 0.23, 7: 0.17 },
    blue:  { 1: 0.04, 2: 0.07, 3: 0.12, 4: 0.19, 5: 0.23, 6: 0.21, 7: 0.14 },
    black: { 1: 0.08, 2: 0.11, 3: 0.16, 4: 0.22, 5: 0.19, 6: 0.14, 7: 0.10 },
    red:   { 1: 0.16, 2: 0.18, 3: 0.20, 4: 0.19, 5: 0.14, 6: 0.08, 7: 0.05 },
    green: { 1: 0.09, 2: 0.13, 3: 0.17, 4: 0.21, 5: 0.18, 6: 0.13, 7: 0.09 }
  }
}
```

When the user answers (say, score = 5), we apply Bayes' theorem:

$$P(c \mid s) = \frac{P(s \mid c) \cdot P(c)}{P(s)}$$

where $P(s) = \sum_c P(s \mid c) \cdot P(c)$ is the marginal probability of the observed score.

In code:

```typescript
function bayesianUpdate(
  distribution: Distribution,
  question: Question,
  score: number
): Distribution {
  const likelihoods = question.scoreProbabilities

  // Calculate marginal probability P(score)
  let marginal = 0
  for (const color of COLORS) {
    marginal += likelihoods[color][score] * distribution[color]
  }

  // Update each color's probability
  const posterior: Distribution = {}
  for (const color of COLORS) {
    posterior[color] = (likelihoods[color][score] * distribution[color]) / marginal
  }

  return posterior
}
```

After each answer, the distribution shifts. Strong signals concentrate probability mass; ambiguous answers spread it out.

## Step 2: Question Selection as Hypothesis Testing

Here's where it gets interesting. Given our current belief distribution, which question should we ask next?

We frame question selection as **active hypothesis testing**. The current distribution is our hypothesis about who you are. We select the next question to *test* that hypothesis - choosing questions where, if our hypothesis is correct, we can predict what you'll answer.

### The Logic

Say we're 80% confident you're Blue. We look for a question where Blue-types give predictable answers - maybe they almost always respond with a 6. If you answer 6, the Bayesian update confirms our belief. If you answer 2, the update shifts probability away from Blue toward archetypes that better explain that answer.

The gist is: questions with **predictable answers** (given current beliefs) are the ones that discriminate between archetypes. By picking questions where our hypothesis makes a strong prediction, we get answers that either confirm or refute it decisively.

### Expected Surprise

We quantify "predictability" using the entropy of the predicted answer distribution. For each candidate question, we compute what answers we'd *expect* given our current beliefs:

$$P(s) = \sum_{c \in \text{colors}} P(s \mid c) \cdot P(c)$$

Then measure the entropy of that distribution:

$$H_{\text{answer}}(q) = -\sum_{s=1}^{7} P(s) \log_2 P(s)$$

Low entropy = we expect a specific answer. High entropy = we have no idea what they'll say.

```typescript
function expectedSurprise(question: Question, distribution: Distribution): number {
  let entropy = 0

  for (let score = 1; score <= 7; score++) {
    let pScore = 0
    for (const color of COLORS) {
      pScore += question.scoreProbabilities[color][score] * distribution[color]
    }

    if (pScore > 0) {
      entropy -= pScore * Math.log2(pScore)
    }
  }

  return entropy
}
```

We select questions that **minimize expected surprise** - the ones where our current hypothesis makes a strong prediction. These are the questions that will either confirm or refute our beliefs most decisively.

## Step 3: Softmax Sampling

We don't greedily pick the minimum-surprise question every time. Instead, we use **temperature-controlled softmax sampling**:

```typescript
function selectNextQuestion(
  questions: Question[],
  currentDist: Distribution,
  temperature: number
): Question {
  // Calculate expected surprise for all unasked questions
  const surprises = questions.map(q => expectedSurprise(q, currentDist))

  // We want LOW surprise, so negate before softmax
  const negSurprises = surprises.map(s => -s)

  // Softmax with temperature
  const maxNeg = Math.max(...negSurprises)
  const expScores = negSurprises.map(s => Math.exp((s - maxNeg) / temperature))
  const sumExp = expScores.reduce((a, b) => a + b, 0)
  const probabilities = expScores.map(e => e / sumExp)

  // Sample from distribution
  const r = Math.random()
  let cumulative = 0
  for (let i = 0; i < questions.length; i++) {
    cumulative += probabilities[i]
    if (r <= cumulative) {
      return questions[i]
    }
  }

  return questions[questions.length - 1]
}
```

Why softmax instead of greedy?

1. **Robustness**: Reduces sensitivity to likelihood calibration errors
2. **Diversity**: Similar users might see different (but still informative) question sequences

Temperature controls the exploration-exploitation tradeoff:
- T → 0: Greedy (always pick lowest expected surprise)
- T → ∞: Random selection

We use a low temperature, making selection nearly deterministic while retaining some stochasticity for robustness.

## Step 4: Preventing Overconfidence

Pure Bayesian updates have a problem: they can become overconfident too quickly. A few consistent answers can push the posterior to 95%+ for one archetype, leaving little room for correction if early answers were noisy or the user was ambivalent.

We address this with **per-step shrinkage** - after each Bayesian update, we pull the distribution slightly back toward uniform:

$$P'(c) = \alpha \cdot \frac{1}{5} + (1 - \alpha) \cdot P(c)$$

where $\alpha$ is a small shrinkage factor. This has two effects:

1. **Prevents runaway confidence**: The posterior can't collapse to a single archetype too quickly
2. **Keeps alternatives alive**: Even if early evidence points strongly to Blue, we maintain some probability mass on other archetypes in case later evidence contradicts

Think of it as epistemic humility baked into the math. We're saying "no matter how confident we are, we might be wrong."

## Step 5: Redundancy Discounting

Another problem: our question pool contains similar questions. If you've already answered three questions about preferring solitude, a fourth solitude question shouldn't shift the posterior as much - it's measuring the same underlying trait.

We maintain a **similarity matrix** between questions. When updating the posterior, we discount based on how much similar information we've already captured:

$$\text{discount} = \prod_{q \in \text{answered}} (1 - \text{similarity}(q, q_{\text{new}}))$$

If the new question is highly similar to previously answered questions, the discount approaches zero and the update is dampened. If it's orthogonal to everything asked so far, the full update applies.

This prevents a cluster of similar questions from dominating the assessment. The test probes multiple facets of personality, not just the ones with the most questions in the pool.

## The Complete Flow

Putting it all together:

```typescript
async function runAssessment(questions: Question[]): Promise<Distribution> {
  let distribution = uniformDistribution()
  const answered: Question[] = []

  for (let i = 0; i < MAX_QUESTIONS; i++) {
    const remaining = questions.filter(q => !answered.includes(q))
    if (remaining.length === 0) break

    // First question is random (uniform prior = all questions equally informative)
    // After that, select by minimum expected surprise
    const nextQuestion = i === 0
      ? randomChoice(remaining)
      : selectNextQuestion(remaining, distribution)

    const score = await presentQuestion(nextQuestion)

    // Bayesian update with redundancy discount
    const discount = redundancyDiscount(nextQuestion, answered)
    const rawPosterior = bayesianUpdate(distribution, nextQuestion, score)
    distribution = blend(rawPosterior, distribution, discount)

    // Per-step shrinkage toward uniform
    distribution = shrinkTowardUniform(distribution, SHRINKAGE_FACTOR)

    answered.push(nextQuestion)
  }

  return distribution
}
```

## Why This Actually Works

### Adaptive Question Ordering

Early in the test, with a uniform prior, any question is about as informative as any other - so we pick randomly. As probability mass concentrates, questions that test the leading hypothesis become more valuable.

A user showing strong Blue tendencies early on won't waste time on Red-vs-Green discrimination questions. The test naturally focuses on confirming or refuting the current best guess.

### Efficient Convergence

By selecting questions that test our current hypothesis, we converge quickly. In practice, we reach stable classifications in around 20 questions instead of 50-100.

### Honest Uncertainty

The final distribution represents genuine epistemic uncertainty. If you're 45% Blue and 40% Black, we show that - because it's true. You're a blend, and forcing a single label would be dishonest.

### Calibration Matters

The whole system depends on accurate likelihood tables. If `scoreProbabilities` are miscalibrated, posteriors will be wrong. We refine these based on anonymous user data.

## Limitations and Trade-offs

**Greedy selection**: We pick questions one at a time based on current beliefs. A globally optimal strategy might ask a less informative question now to set up a more informative one later - but that requires exponential search.

**Calibration-dependent**: Garbage likelihoods = garbage posteriors. We spend significant effort on calibration.

**Independence assumption**: We treat answers as independent given archetype. In reality, there might be order effects or fatigue we don't capture.

## Try It

That's the actual methodology. No hand-waving, no "proprietary AI" bullshit.

Take the test and see if the math holds up for you: **[Start Assessment](/en/new-test)**


