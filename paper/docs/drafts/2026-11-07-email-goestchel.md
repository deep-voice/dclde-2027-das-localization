To: Quentin Goestchel <qgoestch@uw.edu>  (address from the DAS4Whales README; please verify)
Subject: DCLDE 2027 DAS fin whale localization: questions on the baseline

Dear Dr. Goestchel,

I'm Amit Galor, a volunteer at Deep Voice, a non-profit working on machine learning for marine
bioacoustics. We are preparing an entry for the DCLDE 2027 DAS fin whale localization challenge
and have been working with DAS4Whales and the released baseline. Thank you for both.

We asked Dr. Bouffaut the following in October and hoped you might be able to help, since they
concern the baseline:

1. Is there a key (batch, window file, `call_id`) linking the 176 rows of
   `batch1_localizations_with_coords.csv` to the annotated calls? Matching by UTC time is ambiguous.
2. All rows have `z_local = -30.0` (`fix_z=True` in DAS4Whales). Is that the intended L-2
   reference, or is there a source-depth reference for the training calls?
3. Will the test set include arrival-time picks, or only the strain data?

Thank you for your time.

Best regards,
Amit Galor
Deep Voice
