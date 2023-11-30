const fileInput = document.getElementById('imageInput');
const queryfileInput = document.getElementById('queryformFile');

const resultDiv = document.getElementById('result');
const imageContainer = document.getElementById('imageContainer');
let imgbox1 = document.getElementById("imgbox1")
let imgbox2 = document.getElementById("imgbox2")


const toastLive = document.getElementById('liveToast')
const toastLive2 = document.getElementById('liveToast2')

const toastBootstrap = bootstrap.Toast.getOrCreateInstance(toastLive)
const toastBootstrap2 = bootstrap.Toast.getOrCreateInstance(toastLive2)


// Attach the handleFileSelect function to the input change event
fileInput.addEventListener('change', handleFileSelect);
queryfileInput.addEventListener('change', handlequeryFileSelect);

// Initialize the MobileNet model
const featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded);
const featuredataset = [];

toastBootstrap2.show()



// Callback function when the MobileNet model is loaded
function modelLoaded() {
    // Extract features from the image
    console.log("model loaded");
    toastBootstrap.show()
    const fileInputs = document.querySelectorAll('input[type="file"]');

    // Remove the disabled attribute for each file input
    fileInputs.forEach(input => {
        input.removeAttribute('disabled');
    });
}

function gotResults(error, results) {
    console.log("got results !");
    if (error) {
        console.error(error);
    } else {
        // Display the array of logits
        resultDiv.innerText += JSON.stringify(results, null, 2) + '\n\n';
        console.log(results.dataSync());
        results.print();

        // Extract file name from the file input
        const fileName = fileInput.files[featuredataset.length].name;

        // Add logits and file name to the featuredataset
        featuredataset.push({
            features: results.dataSync(),
            name: fileName
        });
    }
}

async function handleFileSelect() {
    console.log("handling files");

    // Check if files are selected
    if (fileInput.files.length > 0) {
        console.log("number of files:", fileInput.files.length);

        for (let i = 0; i < fileInput.files.length; i++) {
            console.log(`image ${i}`);
            const file = fileInput.files[i];
            const img = new Image();
            img.src = window.URL.createObjectURL(file);
            img.className = "card-img-top"; // Add Bootstrap thumbnail class

            // Create a div with class 'col' for each image
            const colDiv = document.createElement('div');

            // Wrap the asynchronous operation in a Promise
            const promise = new Promise((resolve, reject) => {
                img.onload = function () {

                    const fileName = fileInput.files[featuredataset.length].name;




                    const cardDiv = document.createElement('div');
                    cardDiv.className = 'card';
                    cardDiv.classList.add("mb-3")

                    const cardBodyDiv = document.createElement('div');
                    cardBodyDiv.className = 'card-body';

                    const cardText = document.createElement('p');
                    cardText.className = 'card-text';
                    cardText.textContent = fileName;


                    cardBodyDiv.appendChild(cardText);
                    cardDiv.appendChild(img);
                    cardDiv.appendChild(cardBodyDiv);



                    const inDiv = document.createElement('div');
                    inDiv.className = 'col';
                    inDiv.appendChild(cardDiv)


                    // Append the 'col' div to the imageContainer
                    imageContainer.appendChild(inDiv);

                    // Use the MobileNet model to infer features
                    let logits = featureExtractor.infer(img, gotResults);
                    console.log(logits.dataSync());
                    logits.print();


                    // Add logits and file name to the featuredataset
                    featuredataset.push({
                        features: logits.dataSync(),
                        name: fileName,
                        img: img
                    });



                    resolve();
                };
            });

            // Add the Promise to the array
            await promise;
        }

        // All images processed
        console.log('All images processed');

        // Log the feature dataset
        console.log(featuredataset);
    }
}

async function handlequeryFileSelect() {

    // Check if a file is selected
    if (queryfileInput.files.length > 0) {
        imgbox1.innerHTML = ""
        imgbox2.innerHTML = ""

        const file = queryfileInput.files[0];
        const img = new Image();

        img.src = window.URL.createObjectURL(file);
        img.className = "img-thumbnail"; // Add Bootstrap thumbnail class

        img.onload = function () {
            let logits = featureExtractor.infer(img, gotResults);

            imgbox1.appendChild(img);

            // console.log(logits.dataSync())
            // logits.print()

            const mostSimilarImage = findMostSimilarImage(logits.dataSync(), featuredataset);



            const img2 = mostSimilarImage.img.cloneNode(true)
            img2.className = "img-thumbnail"; // Add Bootstrap thumbnail class


            imgbox2.appendChild(img2);


            console.log('Most similar image:', mostSimilarImage);

            document.getElementById("resultrow").classList.remove("d-none")


        };


    }
}

// Function to calculate cosine similarity between two arrays
function cosineSimilarity(array1, array2) {
    // Ensure both arrays have the same length
    if (array1.length !== array2.length) {
        throw new Error('Arrays must have the same length');
    }

    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < array1.length; i++) {
        dotProduct += array1[i] * array2[i];
    }

    // Calculate magnitude
    const magnitude1 = Math.sqrt(array1.reduce((sum, value) => sum + Math.pow(value, 2), 0));
    const magnitude2 = Math.sqrt(array2.reduce((sum, value) => sum + Math.pow(value, 2), 0));

    // Calculate cosine similarity
    if (magnitude1 === 0 || magnitude2 === 0) {
        return 0; // Avoid division by zero
    } else {
        return dotProduct / (magnitude1 * magnitude2);
    }
}

// Function to find the most similar image
function findMostSimilarImage(queryFeatures, imageArray) {
    let maxSimilarity = -1;
    let mostSimilarImage = null;

    for (const image of imageArray) {
        const similarity = cosineSimilarity(queryFeatures, image.features);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            mostSimilarImage = image;
        }
    }

    return mostSimilarImage;
}

