from typing import List

from creator_3d.reconstuctor.actions.action import Action


class Pipeline:
    def __init__(self, steps: List[Action]):
        self.steps = steps

    def run(self, **kwargs):
        # start_params - dict with initial params
        # todo: можно вынести это в отдельный класс параметров, формировать его и получать нужное
        params = kwargs['start_params']
        for step in self.steps:
            step.run(**params)
            params = step.get_result_dict()

    def create_pipeline(self, steps: List[Action]):
        self.steps = steps



