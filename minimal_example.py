import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import re


class ComponentLibraryParser:
    @staticmethod
    def parse_component_library(filePath):
        component_dict = {}
        component_info = None
        name = None

        type_pattern = re.compile(r"type (\w+) (\w+) pwr ([\d.]+) prf ([\d.-]+)")

        with open(filePath, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("@BEGIN COMPONENT"):
                    name = line.split()[2]
                    component_info = {
                        "length": None,
                        "width": None,
                        "flexibility": None,
                        "types": [],
                    }
                elif line.startswith("@END"):
                    if name and component_info:
                        component_dict[name] = component_info
                    component_info = None
                    name = None
                elif component_info is not None:
                    if line.startswith("length"):
                        component_info["length"] = float(line.split()[1])
                    elif line.startswith("width"):
                        component_info["width"] = float(line.split()[1])
                    elif line.startswith("flexibility"):
                        component_info["flexibility"] = int(line.split()[1])
                    else:
                        type_info = type_pattern.findall(line)
                        if type_info:
                            component_type = {
                                "category": type_info[0][0],
                                "type": type_info[0][1],
                                "pwr": float(type_info[0][2]),
                                "prf": float(type_info[0][3]),
                            }
                            component_info["types"].append(component_type)

        return component_dict

    @staticmethod
    def createConfigurationSpace(components):

        # Create a ConfigurationSpace from the component dictionary.
        # Returns: ConfigurationSpace: Configuration space for optimization.

        cs = ConfigurationSpace()
        for category, models in components.items():
            hyperparameter = CategoricalHyperparameter(
                name=category.lower(), choices=models
            )
            cs.add(hyperparameter)

        # Add other hyperparameters
        cs.add(UniformFloatHyperparameter("C", 0.1, 100, default_value=1))
        cs.add(
            CategoricalHyperparameter("kernel", ["linear", "rbf"], default_value="rbf")
        )

        return cs


iris = datasets.load_iris()


def objective_function(config: Configuration, seed: int = 0) -> float:

    # Objective function for SMAC optimization.

    # float: Objective value (e.g., performance metric).

    classifier = SVC(C=config["C"], kernel=config["kernel"], random_state=seed)
    scores = cross_val_score(
        classifier, iris.data, iris.target, cv=5, scoring="accuracy"
    )
    return 1 - np.mean(scores)


def main():
    # Main function to run the SMAC optimization process.
    smac = None  # Initialize smac to None

    try:
        componentLib = "components.lib"
        components = ComponentLibraryParser.parse_component_library(componentLib)
        cs = ComponentLibraryParser.createConfigurationSpace(components)
        scenario = Scenario(cs, deterministic=True, n_trials=200)

        smac = HyperparameterOptimizationFacade(scenario, objective_function)
        bestConfig = smac.optimize()

        print("Best configuration found:", bestConfig)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
