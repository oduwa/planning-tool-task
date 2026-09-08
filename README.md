# Augmented planning decision-making

Thank you for taking the time to complete this assignment. This should take approximately 3-4 hours to complete.

## The Task

Create a decision-support tool that can produce draft decisions on applications for Doncaster Council planning officers. The aim is to reduce the time required to assess housing planning applications from the current average of eight weeks to hours while helping officers make faster, well-supported decisions. In practice this should mean the creation of a system that can produce outcomes in less than 15 minutes. 

The tool must support two outcomes:

- `Approve with conditions`, with the conditions stated
- `Refuse`, with the reasons for refusal stated

## Deliverables

Your PR submission must include:

1. Your completed decision-support tool and any other codebase changes needed to run it.
   > Method, code quality and structure will be evaluated.
2. A completed JSON prediction for each of the five holdout cases (`case-006` to `case-010`) in the root-level `predictions` directory.
3. A report write-up of your method, findings and shortcomings
    - Please explain your choice of method for this task.
    - Please include an honest assessment of your system's strengths and limitations.
    - **You don't have to implement all your ideas, but do communicate them to us.** What would you do if you had more time?

The prediction files are part of the required deliverable. They must use the same `decision`, `reasons`, and `conditions` structure as the expected decision JSON files supplied in `data/doncaster/decisions`. See [Data layout](#data-layout) for the required filenames and format.

> We do not expect you to implement a UI, but we do expect you to think about how your tool could be used in practice.

### Additional Human Workflow Context

- There are national UK policy guidelines and local council policy guidelines which must be applied to the application. 
- Planning officers review the submitted application pack in its entirety, and may make multiple lookups to different relevant policies considering the facts of the case.
- Planning officers need to reason across all types of provided policy materials including policy guidance, diagrams & measurements to be able to make a decision. 
- Understanding which policy section(s) have been relevant to the decision is important for audit and transparency.



## What we provide

The repository contains boilerplate intended to help you get started:

1. A parser that extracts text and images from PDFs
2. A simple evaluator
3. Geospatial tools
4. Supporting data:
   - National policy documents
   - Local policy documents
   - Sample planning applications
   - Expected decisions for the development cases
5. A local setup command for a restricted API key with a small credit balance to access models available through OpenRouter.

You may change as much of the boilerplate as you want to or need. You are also free to use any LLM of your choice for this task. 

## Getting started

### Prerequisites

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python environment and dependency management

### Setup

1. Clone this repository.
2. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) for macOS, Windows, or Linux.
3. From the repository root, run:

   ```bash
   uv run --locked assignment-setup
   ```

4. Enter the passphrase from your recruitment email.

This installs the locked project dependencies and creates a local `.env` file containing your restricted OpenRouter API key. The key is not displayed and the setup does not contact OpenAI DeployCo.

Keep the passphrase and `.env` private. Never commit `.env`. If setup fails, contact the recruitment team.

To check this key's remaining time and spend:

```bash
uv run --locked assignment-status
```

### Using the API key

OpenRouter uses an OpenAI-compatible API. Load `.env` and pass its values to the OpenAI client:

```python
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
)
```

### Data layout

The `data` directory mirrors the planning-decision task:

- `data/national_policy`: national planning policy and legislation
- `data/local_policy`: local planning policy and guidance
- `data/doncaster/applications`: sample planning applications, grouped by `case-###` identifier; each case contains documents in several formats and layouts
- `data/doncaster/decisions`: expected decisions in JSON format for cases where the real outcome is provided
- `predictions`: placeholder files for your predicted outcomes on holdout cases

The policy PDFs are official source documents. Their licensing and attribution details are recorded in [`data/LICENSING.md`](data/LICENSING.md).

The sample application packages are inspired by real planning scenarios. Identifying details and case characteristics in the supplied sample documents are fictional.

Cases `case-006` to `case-010` are holdout cases. Their correct decisions are not included in this repository. For these cases, generate your own prediction files in the root-level `predictions` directory:

- `predictions/case-006-prediction.json`
- `predictions/case-007-prediction.json`
- `predictions/case-008-prediction.json`
- `predictions/case-009-prediction.json`
- `predictions/case-010-prediction.json`

Each prediction must be valid JSON with the same structure as the expected decisions in `data/doncaster/decisions`:

```json
{
  "decision": "approve",
  "reasons": [
    "Explain the policy and evidence basis for the recommendation."
  ],
  "conditions": [
    "State each proposed condition for an approval."
  ]
}
```

Set `decision` to either `"approve"` or `"refuse"`. Use an empty `conditions` array for a refusal. We will compare these predictions with the real outcomes after submission.

Before submitting, validate the prediction files from the repository root:

```bash
uv run python scripts/validate_submission.py
```

The validator checks that all five required JSON files exist, use exactly the expected fields, contain a valid decision and non-empty reasons, and include conditions for approvals only. The same validation runs automatically when you open or update a pull request to `main`.

### PDF parser

The PDF parser is in `src/parser`.

To parse every PDF under `data`, run:

```bash
uv run parser
```

The command writes extracted JSON under `data/extracted` and preserves the source directory structure.

#### Document data model

Each extracted PDF becomes a `PdfDocument` containing:

- `PdfPage`: one object per page
- `PdfTextBlock`: extracted text for a page
- `PdfImageBlock`: extracted image data for a page

The following helpers return content in document order:

```python
texts = doc.get_texts()  # list[str]
images = doc.get_images()  # list[bytes]
```

### Geospatial tools

The geospatial tools in [`src/tools`](src/tools) provide a deterministic lookup
from the fictional postcodes in the supplied applications to relevant planning
features, including flood-risk zones, conservation areas, green belt, and
heritage constraints.

For simplicity in this time-boxed task, the postcode-to-feature mappings are
hardcoded rather than retrieved from external geospatial services. The lookup is
therefore limited to the supplied assignment cases and must not be treated as
authoritative planning data. Unknown postcodes are reported explicitly instead
of being treated as locations with no constraints.

```python
from tools.geospatial import postcode_lookup

constraints = postcode_lookup("ZZ46 0BG")
```

See the [geospatial tools README](src/tools/README.md) for the response schema,
examples, limitations, etc.

### Simple evaluator

The simple evaluator is in `src/evaluator`. It provides a starting point for assessing decision accuracy and reasoning alignment.

## Submission

Submit your completed work as **one clean pull request**. Do not open separate pull requests for different parts of the assignment or for later fixes.

Before opening the pull request:

1. Complete your solution, all five prediction files and `REPORT.md`.
2. Run the submission validator:

   ```bash
   uv run python scripts/validate_submission.py
   ```

3. Review your changes and remove temporary files, credentials and anything unrelated to your solution.
4. Commit and push all of your completed work to one branch.

Open one pull request with the following details:

- **Base branch:** `main`
- **Title:** `Submission` — exactly as written
- **Status:** Ready for review, not draft

Complete the checklist in the pull request description and wait for the **Validate submission** check to pass. If you need to make a correction, push it to the same branch so it updates the same pull request.

Send that pull request URL in response to the recruitment email. Do not merge or close the pull request.

## Use of AI tools

Please disclose whether you used AI tools to generate code or support any other part of your submission, and explain how and where you used them. AI-powered IDEs and coding assistants are acceptable; you remain responsible for understanding and owning the submitted work.

You may include any agent instruction files that you created or used.

As in professional work, follow these principles:

- Make sure you understand any code before including it; you may be asked to explain it.
- Do not submit code that you do not understand.
- You are responsible for the behaviour, quality, and stability of your solution.
- AI-generated code can support, but not replace, your understanding and ownership.

### Disclaimer

This assignment is representative of the kinds of problems OpenAI DeployCo. works on, but it is a standalone recruitment exercise rather than a real client, council, or government project.

OpenAI DeployCo. is not affiliated with, acting on behalf of, or endorsed by the UK Government, City of Doncaster Council, or any other public-sector body referenced in this repository. Public-sector documents are included solely to provide realistic context, subject to the terms described in [`data/LICENSING.md`](data/LICENSING.md). Any assessments, recommendations, or decisions generated during the exercise are illustrative and must not be treated as official planning advice or decisions.
