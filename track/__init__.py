import os
import neptune
from neptune import management


class TrackBase:
    def __init__(self) -> None:
        pass


class Neptune(TrackBase):
    def __init__(self, project_name, config, tags, project_key=None) -> None:
        super().__init__()
        project = f'doveppp/{project_name}'
        if project not in management.get_project_list(api_token=self.api_token):
            management.create_project(
                project,
                key=project_key,
            )
        run = neptune.init_run(
            project=project,
            api_token=self.api_token,
            tags=tags,
            # mode='offline',
            capture_hardware_metrics=False,
            capture_stderr=False,
            capture_stdout=False,
            capture_traceback=False,
        )
        run["config"] = config
        run["model_files"].upload_files("models/*.py")
        if os.environ.get('RUN_COMMENT'):
            run['run_commit'] = os.environ['RUN_COMMENT']
        self.run = run

    def finish(self):
        pass

    def log(self, dic):
        for k, v in dic.items():
            self.run[k].append(v)
        # self.run["data"] = data


if __name__ == "__main__":
    tracker = Neptune('test', {}, ["a", "b"])
    for i in range(30):
        print(i)
        tracker.log({'t': i})
