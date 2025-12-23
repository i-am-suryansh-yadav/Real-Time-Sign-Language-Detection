// ui/static/script.js (Updated to handle camera launch, detection toggle, and stop without freeze)
document.addEventListener('DOMContentLoaded', () => {
    const launchButtons = document.querySelectorAll('#launch-header, #launch-hero');
    const stopButton = document.getElementById('stop-camera');
    const toggleButton = document.getElementById('toggle-detection');
    const videoFeed = document.getElementById('video-feed');
    const cameraSection = document.getElementById('camera-section');

    let detecting = false;

    // Launch Camera
    launchButtons.forEach(button => {
        button.addEventListener('click', () => {
            fetch('/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'camera started') {
                        videoFeed.src = '/video_feed';
                        cameraSection.classList.remove('hidden');
                        cameraSection.scrollIntoView({ behavior: 'smooth' });
                        // Reset detection toggle
                        detecting = false;
                        toggleButton.textContent = 'Start Detection';
                        toggleButton.classList.remove('stop');
                    }
                })
                .catch(error => console.error('Error starting camera:', error));
        });
    });

    // Toggle Detection
    toggleButton.addEventListener('click', () => {
        fetch('/toggle_detection', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                detecting = data.detecting;
                if (detecting) {
                    toggleButton.textContent = 'Stop Detection';
                    toggleButton.classList.add('stop');
                } else {
                    toggleButton.textContent = 'Start Detection';
                    toggleButton.classList.remove('stop');
                }
            })
            .catch(error => console.error('Error toggling detection:', error));
    });

    // Stop Camera
    stopButton.addEventListener('click', () => {
        fetch('/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'camera stopped') {
                    videoFeed.src = '';  // Clear src to stop stream
                    videoFeed.load();    // Force reload to prevent freeze
                    cameraSection.classList.add('hidden');
                    // Reset detection
                    detecting = false;
                    toggleButton.textContent = 'Start Detection';
                    toggleButton.classList.remove('stop');
                }
            })
            .catch(error => console.error('Error stopping camera:', error));
    });
});