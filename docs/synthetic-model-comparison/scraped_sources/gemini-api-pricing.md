# Google Gemini API Pricing

Source URL: https://ai.google.dev/gemini-api/docs/pricing
Scraped: 2026-07-14

## Pricing Tiers

Google offers three service levels: **Free** (limited models, free tokens, data used for improvement), **Paid** (higher rate limits, advanced features, data not used for improvement), and **Enterprise** (custom support, security, provisioned throughput).

## Text Models - Key Pricing

**Gemini 3.5 Flash** (Standard inference):
- Free: All tokens free
- Paid: $1.50/M input tokens, $9.00/M output tokens

**Gemini 3.1 Flash-Lite** (most cost-efficient):
- Free: All tokens free
- Paid: $0.25/M input (text/image/video), $1.50/M output

**Gemini 2.5 Flash-Lite** (smallest model):
- Free: All tokens free
- Paid: $0.10/M input (text/image/video), $0.40/M output

## Batch & Optimization Discounts

Batch processing offers 50% cost reduction. Example (Gemini 3.5 Flash Batch):
- Input: $0.75/M tokens
- Output: $4.50/M tokens

Flex inference available at same batch rates. Priority inference costs 1.8x standard rates.

## Image Generation Pricing

**Gemini 2.5 Flash Image**:
- Input: $0.30/M tokens
- Output: $0.039 per 1K resolution image (Standard)
- Batch: $0.0195 per image (50% discount)

**Gemini 3.1 Flash Image**:
- Input: $0.50/M tokens
- Output: $60/M tokens ($0.045-$0.151 per image depending on resolution)

## Video & Audio Models

**Veo 3.1** (video generation): $0.05-$0.60 per second depending on quality/speed tier

**Gemini 3.1 Flash Live**: $3.00/M audio input or $0.005/min; $12.00/M audio output or $0.018/min

## Embedding Models

**Gemini Embedding 2** (multimodal):
- Text: $0.20/M tokens
- Images: $0.45/M tokens ($0.00012 per image)

## Tool Pricing

Google Search: 5,000 prompts/month free (Gemini 3), then $14/1,000 queries. Google Maps: Similar structure at $25/1,000 queries.
