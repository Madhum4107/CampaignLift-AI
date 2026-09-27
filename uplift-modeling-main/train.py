from src.models.pipeline import UpliftPipeline
if __name__ == "__main__":
    results,best=UpliftPipeline().fit(); print("Best model:",best); print(results)
