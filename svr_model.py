from flask import Flask, request, render_template, redirect, url_for
from txt2img_model import create_pipeline, txt2img

app = Flask(__name__)

IMAGE_PATH = "static/output.jpg"

model_list = [
    "nota-ai/bk-sdm-small",
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "prompthero/openjourney",
    "hakurei/waifu-diffusion",
    "stabilityai/stable-diffusion-2-1",
    "dreamlike-art/dreamlike-photoreal-2.0",
]


@app.route("/", methods=["GET"])
def index():
    if request.method == "GET":
        return render_template("index.html", model_list=model_list)


@app.route("/nota-ai/bk-sdm-small", methods=["GET", "POST"])
def nota_ai():
    if request.method == "GET":
        selected_model = "nota-ai/bk-sdm-small"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "nota-ai/bk-sdm-small"
        pipeline = create_pipeline(model_list[0])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/CompVis/stable-diffusion-v1-4", methods=["GET", "POST"])
def CompVis():
    if request.method == "GET":
        selected_model = "CompVis/stable-diffusion-v1-4"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "CompVis/stable-diffusion-v1-4"
        pipeline = create_pipeline(model_list[1])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/runwayml/stable-diffusion-v1-5", methods=["GET", "POST"])
def runwayml():
    if request.method == "GET":
        selected_model = "runwayml/stable-diffusion-v1-5"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "runwayml/stable-diffusion-v1-5"
        pipeline = create_pipeline(model_list[2])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/prompthero/openjourney", methods=["GET", "POST"])
def prompthero():
    if request.method == "GET":
        selected_model = "prompthero/openjourney"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "prompthero/openjourney"
        pipeline = create_pipeline(model_list[3])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/hakurei/waifu-diffusion", methods=["GET", "POST"])
def hakurei():
    if request.method == "GET":
        selected_model = "hakurei/waifu-diffusion"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "hakurei/waifu-diffusion"
        pipeline = create_pipeline(model_list[4])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/stabilityai/stable-diffusion-2-1", methods=["GET", "POST"])
def stabilityai():
    if request.method == "GET":
        selected_model = "stabilityai/stable-diffusion-2-1"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "stabilityai/stable-diffusion-2-1"
        pipeline = create_pipeline(model_list[5])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


@app.route("/dreamlike-art/dreamlike-photoreal-2.0", methods=["GET", "POST"])
def dreamlike():
    if request.method == "GET":
        selected_model = "dreamlike-art/dreamlike-photoreal-2.0"
        return render_template(
            "model.html", model_list=model_list, selected_model=selected_model
        )
    else:
        selected_model = "dreamlike-art/dreamlike-photoreal-2.0"
        pipeline = create_pipeline(model_list[6])
        user_input = request.form["prompt"]
        print("Start gen...")
        im = txt2img(user_input, pipeline)
        print("Finish gen...")
        im.save(IMAGE_PATH)

        return render_template(
            "model.html",
            model_list=model_list,
            image_url=IMAGE_PATH,
            selected_model=selected_model,
        )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8888, use_reloader=False)
