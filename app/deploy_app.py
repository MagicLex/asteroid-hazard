"""Deploy the Asteroid Doomsday-o-meter Streamlit app on Hopsworks.

Clones python-app-pipeline, pins the model's training stack (so the pickle
loads), then creates and starts the app. The code already lives on HopsFS (the
FUSE home is the project dataset), so there is nothing to upload. Paths
self-derive from __file__ so this is portable across users.
"""
from pathlib import Path

import hopsworks

HERE = Path(__file__).resolve().parent
# /hopsfs/Users/<user>/...  ->  project-relative  Users/<user>/...
REL = "Users/" + str(HERE).split("/hopsfs/Users/", 1)[1]

APP_NAME = "asteroid-doomsday"
ENV_NAME = "asteroid-app-env"


def main():
    project = hopsworks.login()
    env_api = project.get_environment_api()
    apps = project.get_app_api()

    # 1. env: clone base + pin the pickle's training versions
    env = env_api.get_environment(ENV_NAME)
    if env is None:
        print(f"cloning python-app-pipeline -> {ENV_NAME} ...", flush=True)
        env = env_api.create_environment(
            ENV_NAME, base_environment_name="python-app-pipeline")
    print("installing app-requirements.txt (minutes) ...", flush=True)
    env.install_requirements(f"{REL}/app-requirements.txt", await_installation=True)
    print("env ready", flush=True)

    # 2. create + start (code already on HopsFS via the FUSE home)
    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(
            name=APP_NAME, app_path=f"{REL}/app.py",
            environment=ENV_NAME, memory=2048, cores=1.0)
    app.run(await_serving=True)
    print("state:", app.state, "serving:", app.serving)
    print("URL:", app.get_url())


if __name__ == "__main__":
    main()
