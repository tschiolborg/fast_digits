const canvas = document.getElementById("drawing_board")
const clear = document.getElementById("clear")
const predict = document.getElementById("predict")
const prediction = document.getElementById("prediction")
const context = canvas.getContext("2d")

// predict

function predict_image() {
    canvas.toBlob((blob) => {
        let file = new File([blob], "file.jpg", { type: "image/jpeg" })

        var formData = new FormData()
        formData.append("image", file)

        const requestOptions = {
            method: 'POST',
            body: formData
        }
        fetch('/predict', requestOptions)
            .then(response => response.json())
            .then(function (response) {
                console.log(response)
                prediction.innerHTML = response["label"]
            })
    })
}
predict.addEventListener("click", predict_image)


// draw

canvas.height = 300
canvas.width = 300
context.lineWidth = 5
context.lineCap = "round"

let draw = false
let x = null
let y = undefined

clear.addEventListener("click", () => {
    context.clearRect(x = 0, y = 0, w = canvas.width, h = canvas.height)
    prediction.innerHTML = ""
});

canvas.addEventListener("mousedown", e => {
    draw = true
    x = e.offsetX
    y = e.offsetY
});

canvas.addEventListener("mouseup", () => {
    draw = false;
    x = null
    y = null
});

canvas.addEventListener("mousemove", e => {
    if (!draw) {
        return
    }
    const x2 = e.offsetX
    const y2 = e.offsetY

    context.beginPath()
    context.moveTo(x, y)
    context.lineTo(x2, y2)
    context.stroke()

    x = x2
    y = y2
});
