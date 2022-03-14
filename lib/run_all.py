import multiprocessing
# without this running explainer.shap_values locked...
multiprocessing.set_start_method("spawn", force=True)
import lib.real_examples as real_examples
import lib.compute_instance as compute_instance
from multiprocessing import Process


def run_instance(dataset, model_type):
    NAME, X, model = dataset(model_type)
    print("*"*10, NAME, " ", model_type, "*"*5)
    compute_instance.compute_instance(NAME, X, model, model_type)

def main():
    processes = []
    for dataset in [real_examples.get_boston, real_examples.get_nhanes, real_examples.get_health_insurance, real_examples.get_flights]:
        for model_type in ['GBDT', 'DT']:
          p = Process(target=run_instance, args=(dataset, model_type))
          p.start()
          processes.append(p)
    
    for p in processes:
      p.join()


if __name__ == '__main__':
  main()