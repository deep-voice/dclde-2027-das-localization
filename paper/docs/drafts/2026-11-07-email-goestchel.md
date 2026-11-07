To: Quentin Goestchel <qgoestch@uw.edu>  (address from the DAS4Whales README; please verify)
Subject: DCLDE 2027 DAS fin whale localization: questions on the baseline localizations

Dear Dr. Goestchel,

My name is Amit Galor, and I'm a volunteer at Deep Voice, a non-profit working on machine learning
for bioacoustics and marine conservation. We are preparing an entry for the DCLDE 2027 DAS fin whale
localization challenge (L-1 and L-2), and we have been working closely with your DAS4Whales package
and the baseline output released with the training data. Thank you for making both available.

We wrote to Dr. Bouffaut about the questions below in October. We know organizers are busy, and since
they concern the baseline you produced, we hoped you might be able to help:

1. Call key for the baseline localizations. `batch1_localizations_with_coords.csv` (176 rows) has no
   call identifier, and its `id` looks like a loop index. Matching rows to the annotated calls by UTC
   time is ambiguous at about one call every 4 s, even using the `sensor` column. Is there a key
   (batch, window file and `call_id`) linking each row to its annotated call?
2. Depth in the baseline. All 176 rows have `z_local = -30.0`, which matches the `fix_z=True`,
   z0 = -30 m setting in DAS4Whales. Is that the intended reference for L-2, or is there a
   source-depth reference for the training calls?
3. The test set. The challenge page describes it as 10 minutes of labeled data without positions.
   Will it include arrival-time picks like the training annotations, or only the strain data?

In case it is useful, one observation from our side: re-picking calls on the dense channels gives
picks consistent to about 13 ms between neighbouring channels. Yet against a straight-ray,
constant-speed (1480 m/s) model the residuals keep a smooth, km-scale structure of about 60 ms. That
structure repeats for calls from the same position and differs between positions a few km apart.
We would be glad to hear whether this matches your experience, and to share details.

Thank you for your time.

Best regards,
Amit Galor
Deep Voice
