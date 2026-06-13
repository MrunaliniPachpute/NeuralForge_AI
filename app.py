import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecret"
app.config["UPLOAD_FOLDER"] = "static_uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Force CPU for Render
device = torch.device("cpu")

encoder = None
decoder = None


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


def load_models():
    global encoder, decoder

    if encoder is None:
        print("Loading encoder...")
        encoder = VGGEncoder("vgg_normalised.pth").to(device)
        encoder.eval()
        print("Encoder loaded.")

    if decoder is None:
        print("Loading decoder...")
        decoder = Decoder().to(device)
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        DECODER_PATH = os.path.join(
            BASE_DIR,
            "experiment",
            "final_28epochmodel",
            "decoder_3.pth"
        )

        print("Decoder path:", DECODER_PATH)

        decoder.load_state_dict(
            torch.load(
                DECODER_PATH,
                map_location=device
            )
        )

        decoder.eval()
        print("Decoder loaded.")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_tensor = (
        transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        transform(style_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        content_feats = encoder(content_tensor)[-1]
        style_feats = encoder(style_tensor)[-1]

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        stylized_image = decoder(stylized_feats)

        return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        if form.content.data and form.content.data.filename:

            if allowed_file(form.content.data.filename):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                form.content.data.save(
                    os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        content_filename
                    )
                )

                form.content_path.data = content_filename

        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:

            if allowed_file(form.style.data.filename):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                form.style.data.save(
                    os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        style_filename
                    )
                )

                form.style_path.data = style_filename

        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:

            try:

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")

                alpha = float(form.alpha.data)

                load_models()

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = (
                    "stylized_" + content_filename
                )

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

            except Exception as e:
                error = str(e)
                print("ERROR:", e)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        error=error,
        content_image=content_filename,
        style_image=style_filename
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(
        "examples",
        filename
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )