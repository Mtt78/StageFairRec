# Publish the source repository

Create an empty GitHub repository named `StageFairRec`, then run these commands
inside the extracted `StageFairRec` directory. Replace YOUR_ACCOUNT explicitly:

```bash
git init -b main
git add .
git commit -m "Initial paper-based StageFairRec implementation"
git remote add origin https://github.com/YOUR_ACCOUNT/StageFairRec.git
git push -u origin main
```

Alternatively upload the contents through the GitHub website. Include `.github/`
and `.gitignore`; do not upload the outer ZIP as the only repository file.
The repository is MIT-licensed. Correct the collective attribution in LICENSE and
CITATION.cff when confirmed authorship metadata is available.

For an author experiment release, first reconcile the documented assumptions with
your original data/code, run the real experiments, and then replace the reconstruction
status with an accurate description of what was validated. Do not copy paper table
values into generated result files. Update the manuscript's code URL after this new
repository actually exists and is accessible. No remote repository was created or
published as part of generating this source package.
