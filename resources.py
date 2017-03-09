import os
import requests

resources_dir = 'resources'

def _cached_resource(url):
    filepath = os.path.join(resources_dir, url.rsplit('/', 1)[-1])

    if not os.path.exists(resources_dir):
        os.mkdir(resources_dir)

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    else:
        content = requests.get(url).text
        with open(filepath, 'w') as f:
            f.write(content)
        return content

def get_iris_dataset():
    return _cached_resource('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'),\
            _cached_resource('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names')

if __name__ == "__main__":
    get_iris_dataset()
