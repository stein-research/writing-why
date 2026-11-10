# Data

Neither dataset is stored here; download them into these folders (or point the environment variables at copies
elsewhere) and check the sha256.

## TinyStories V2 (GPT-4 splits) -> `data/tinystories/`

From https://huggingface.co/datasets/roneneldan/TinyStories (`resolve/main/<file>`). Override the location with
`WRITING_WHY_TINYSTORIES=/path/to/TinyStoriesV2-GPT4-valid.txt` (the training file is read from the same folder).

| file | bytes | sha256 |
|---|---|---|
| `TinyStoriesV2-GPT4-valid.txt` | 22,502,601 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |
| `TinyStoriesV2-GPT4-train.txt` | 2,227,753,162 | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |

Licence: CDLA-Sharing-1.0 (the dataset card). Reference: R. Eldan and Y. Li, "TinyStories: How Small Can Language
Models Be and Still Speak Coherent English?", arXiv:2305.07759 (2023). The word vocabulary built from its training
stories is shipped as `results/vocab_tinystories.json` (derived data, CDLA-Sharing-1.0), so the models sample and
explain without the 2.2 GB training file; training and the test windows need the files above.

## WikiText-103 (raw, v1) -> `data/wikitext103/`

From https://huggingface.co/datasets/Salesforce/wikitext, config `wikitext-103-raw-v1` (Parquet). Override with
`WRITING_WHY_WIKITEXT=/path/to/folder`.

    74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212  train-00000-of-00002.parquet
    ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720  train-00001-of-00002.parquet
    204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c  validation-00000-of-00001.parquet
    5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91  test-00000-of-00001.parquet

Licence: Wikipedia text, CC BY-SA and GFDL. Reference: S. Merity, C. Xiong, J. Bradbury and R. Socher, "Pointer
Sentinel Mixture Models", arXiv:1609.07843 (2016). Its word vocabulary is shipped as `results/vocab_wikitext103.json`
(derived data, CC BY-SA).
