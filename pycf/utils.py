import IPython.display
import datetime
from PIL import Image
from io import BytesIO

from matplotlib import pyplot as plt



def show_graph(graph_def, frame_size=(900, 600)):
    """Visualize TensorFlow graph.
    Credit goes to the University of Edinburgh Team.
    Namely the team handling the machine learning practical repository at:
    https://github.com/CSTR-Edinburgh/mlpractical
    """
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(height=frame_size[1], data=repr(str(graph_def)), id='graph'+timestamp)
    iframe = """
        <iframe seamless style="width:{width}px;height:{height}px;border:0" srcdoc="{src}"></iframe>
    """.format(width=frame_size[0], height=frame_size[1] + 20, src=code.replace('"', '&quot;'))
    IPython.display.display(IPython.display.HTML(iframe))


def display_image(image, mode='RGB', fmt='png'):
    mfile = BytesIO()
    Image.fromarray(image.astype('int8'), mode).save(mfile, fmt)
    IPython.display.display(IPython.display.Image(data=mfile.getvalue()))


def save_image(image, filename, mode='RGB'):
    Image.fromarray(image.astype('int8'), mode).save('images/' + filename)
