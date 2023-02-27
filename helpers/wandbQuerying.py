import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("slahmer/costoflearning-icc23/b6pk0z9k")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")