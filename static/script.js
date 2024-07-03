document.addEventListener('DOMContentLoaded', function() {
    const videoSelect = document.getElementById('videoSelect');
    const loadVideoButton = document.getElementById('loadVideoButton');
    const videoPreview = document.getElementById('videoPreview');
    const processButton = document.getElementById('processButton');
    const resultText = document.getElementById('resultText');
    const spinner = document.getElementById('spinner');

    // Fetch and populate video list
    fetch('/list_videos')
        .then(response => response.json())
        .then(data => {
            data.videos.forEach(video => {
                const option = document.createElement('option');
                option.value = video;
                option.textContent = video;
                videoSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Error:', error));

    // Load selected video into video player
    loadVideoButton.addEventListener('click', function() {
        const selectedVideo = videoSelect.value;
        if (selectedVideo) {
            const videoUrl = `/videos/${selectedVideo}`;
            videoPreview.src = videoUrl;
            videoPreview.style.display = 'block';
        } else {
            alert('Please select a video first.');
        }
    });

    // Process the selected video
    processButton.addEventListener('click', function() {
        const selectedVideo = videoSelect.value;
        if (selectedVideo) {
            resultText.textContent = '';
            spinner.style.display = 'block'; // Show the spinner
            const formData = new FormData();
            formData.append('video', selectedVideo);

            fetch('/process_selected', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                resultText.textContent = `Result: ${data.result.label} (Accuracy: ${data.result.accuracy.toFixed(2)}%)`;
                spinner.style.display = 'none'; // Hide the spinner
            })
            .catch(error => {
                console.error('Error:', error);
                resultText.textContent = 'Error processing video';
                spinner.style.display = 'none'; // Hide the spinner
            });
        } else {
            alert('Please select a video first.');
        }
    });
});
