HNGTN Training Data

the Training_Data.txt file is a preprocessed subset of the WikiText-2 training dataset,
introduced by Merity et al. in “Pointer Sentinel Mixture Models” (2016).
WikiText-2 was derived from Wikipedia articles. This version was truncated,
tokenized, and divided into chunks for the HNGTN experiment.
The text remains subject to the applicable WikiText/Wikipedia
Creative Commons Attribution-ShareAlike and GFDL licensing terms.

Changes made:
- selected the first 13,714 lines;
- retained 763,000 tokens;
- divided the stream into 763 chunks of 1,000 tokens.
