# OpenAI Image Generation Pricing Documentation

**Source URLs:**
- https://developers.openai.com/api/docs/pricing#image-tokens
- https://developers.openai.com/api/docs/guides/image-generation#calculating-costs

**Scraped:** 2026-07-14

---

## Source 1: Pricing Documentation (Image Tokens)

### Image Generation Pricing

#### Models and Standard Pricing

OpenAI offers three image generation models with distinct pricing structures:

| Model | Image Input | Image Cached | Text Input | Text Cached | Output |
|-------|------------|--------------|-----------|------------|--------|
| gpt-image-2 | $8.00 | $2.00 | $5.00 | $1.25 | $30.00 |
| gpt-image-1.5 | $8.00 | $2.00 | $5.00 | $1.25 | $10.00 |
| gpt-image-1-mini | $2.50 | $0.25 | $2.00 | $0.20 | $8.00 |

All rates are per 1M tokens.

#### Batch Processing Discounts

Batch mode reduces costs by 50% across all image generation models:

| Model | Image Input | Image Cached | Text Input | Text Cached | Output |
|-------|------------|--------------|-----------|------------|--------|
| gpt-image-2 | $4.00 | $1.00 | $2.50 | $0.625 | $15.00 |
| gpt-image-1.5 | $4.00 | $1.00 | $2.50 | $0.63 | $5.00 |
| gpt-image-1-mini | $1.25 | $0.13 | $1.00 | $0.10 | $4.00 |

#### Key Notes

The documentation references "a calculator in the image generation guide" for detailed cost estimates based on specific parameters, though dimensional options and quality tiers aren't itemized in the pricing table itself.

---

## Source 2: Image Generation Cost Calculation Guide

### GPT-Image-2 Pricing Structure

The `gpt-image-2` model uses a simplified output token system. According to the documentation, "use the calculator to estimate output tokens from the requested `quality` and `size`."

#### Output Token Calculation for GPT-Image-2

The pricing varies based on two parameters:

**Quality Settings:**
- Low
- Medium  
- High

**Size Dimensions:** The model accepts "any resolution in the `size` parameter" that meets these constraints:
- Maximum edge: ≤3,840px
- Both edges: multiples of 16px
- Aspect ratio: ≤3:1
- Total pixels: 655,360–8,294,400

**Example Pricing Table (1024×1024):**

| Quality | Cost per Image |
|---------|----------------|
| Low | $0.006 |
| Medium | $0.053 |
| High | $0.211 |

### Legacy Models (GPT-Image-1.5, 1, 1-Mini)

These earlier versions employed token-based pricing where "larger image sizes and higher quality settings result in more tokens."

**Sample Output Tokens (1024×1024):**

| Model | Low | Medium | High |
|-------|-----|--------|------|
| GPT Image 1.5 | 272 | 1,056 | 4,160 |
| GPT Image 1 | 272 | 1,056 | 4,160 |
| GPT Image 1 Mini | 272 | 1,056 | 4,160 |

### Total Cost Components

The complete cost includes:
1. Input text tokens (prompt)
2. Input image tokens (if using edits)
3. Output image tokens

"Refer to the [pricing page](/api/docs/pricing#image-generation) for current text and image token prices."

### Streaming Cost Consideration

When using partial image streaming, "each partial image will incur an additional 100 image output tokens."
