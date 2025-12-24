// Enhanced SignSync JavaScript - OPTIMIZED
document.addEventListener('DOMContentLoaded', () => {
    const launchButtons = document.querySelectorAll('#launch-header, #launch-hero');
    const stopButton = document.getElementById('stop-camera');
    const toggleButton = document.getElementById('toggle-detection');
    const clearWordButton = document.getElementById('clear-word');
    const videoFeed = document.getElementById('video-feed');
    const videoContainer = document.getElementById('video-container');
    const cameraSection = document.getElementById('camera-section');
    const wordOutput = document.getElementById('word-output');
    const statusText = document.getElementById('status-text');
    const fpsText = document.getElementById('fps-text');
    const navLinks = document.querySelectorAll('.nav-link');
    const demoLinks = document.querySelectorAll('.demo-link');
    const pages = document.querySelectorAll('.page');

    let detecting = false;
    let wordUpdateInterval = null;

    // Page Navigation
    function showPage(pageName) {
        pages.forEach(page => page.classList.remove('active'));
        
        const targetPage = document.getElementById(`${pageName}-page`);
        if (targetPage) {
            targetPage.classList.add('active');
        }

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.dataset.page === pageName) {
                link.classList.add('active');
            }
        });

        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (pageName !== 'home') {
            cameraSection.classList.add('hidden');
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            showPage(link.dataset.page);
        });
    });

    demoLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            showPage('demo');
        });
    });

    // OPTIMIZED: Update word display periodically
    function startWordUpdates() {
        if (wordUpdateInterval) return;
        
        wordUpdateInterval = setInterval(async () => {
            if (detecting) {
                try {
                    const response = await fetch('/get_word');
                    const data = await response.json();
                    wordOutput.textContent = data.word || '...';
                } catch (error) {
                    console.error('Word update error:', error);
                }
            }
        }, 500); // Update every 500ms
    }

    function stopWordUpdates() {
        if (wordUpdateInterval) {
            clearInterval(wordUpdateInterval);
            wordUpdateInterval = null;
        }
        wordOutput.textContent = '...';
    }

    // OPTIMIZED: Launch Camera with loading state
    launchButtons.forEach(button => {
        button.addEventListener('click', async () => {
            showPage('home');
            
            button.disabled = true;
            button.textContent = 'Starting...';
            
            try {
                const response = await fetch('/start', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'camera started') {
                    // Set video source
                    videoFeed.src = '/video_feed?t=' + new Date().getTime();
                    
                    // Show camera section immediately
                    cameraSection.classList.remove('hidden');
                    
                    // Scroll to camera section
                    setTimeout(() => {
                        cameraSection.scrollIntoView({ 
                            behavior: 'smooth',
                            block: 'center'
                        });
                    }, 100);
                    
                    // Reset states
                    detecting = false;
                    toggleButton.innerHTML = '<span class="btn-glow"></span><span class="btn-icon">●</span> Start Detection';
                    toggleButton.classList.remove('stop');
                    videoContainer.classList.remove('detecting');
                    statusText.textContent = 'Camera Ready';
                    wordOutput.textContent = '...';
                }
            } catch (error) {
                console.error('Camera start error:', error);
                alert('Failed to start camera. Please check permissions and try again.');
            } finally {
                button.disabled = false;
                button.innerHTML = '<span class="btn-glow"></span> Launch Camera';
            }
        });
    });

    // Toggle Detection with visual feedback
    toggleButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/toggle_detection', { method: 'POST' });
            const data = await response.json();
            detecting = data.detecting;
            
            if (detecting) {
                toggleButton.innerHTML = '<span class="btn-glow"></span><span class="btn-icon">■</span> Stop Detection';
                toggleButton.classList.add('stop');
                videoContainer.classList.add('detecting');
                statusText.textContent = 'Detecting';
                statusText.style.color = '#00ff00';
                startWordUpdates();
            } else {
                toggleButton.innerHTML = '<span class="btn-glow"></span><span class="btn-icon">●</span> Start Detection';
                toggleButton.classList.remove('stop');
                videoContainer.classList.remove('detecting');
                statusText.textContent = 'Paused';
                statusText.style.color = '#ff6b6b';
                stopWordUpdates();
            }
        } catch (error) {
            console.error('Toggle detection error:', error);
            alert('Failed to toggle detection. Please try again.');
        }
    });

    // Clear Word
    clearWordButton.addEventListener('click', async () => {
        try {
            await fetch('/clear_word', { method: 'POST' });
            wordOutput.textContent = '...';
            console.log('Word cleared');
        } catch (error) {
            console.error('Clear word error:', error);
        }
    });

    // Stop Camera
    stopButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/stop', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'camera stopped') {
                videoFeed.src = '';
                videoFeed.load();
                
                cameraSection.style.opacity = '0';
                setTimeout(() => {
                    cameraSection.classList.add('hidden');
                    cameraSection.style.opacity = '1';
                }, 300);
                
                detecting = false;
                toggleButton.innerHTML = '<span class="btn-glow"></span><span class="btn-icon">●</span> Start Detection';
                toggleButton.classList.remove('stop');
                videoContainer.classList.remove('detecting');
                statusText.textContent = 'Stopped';
                statusText.style.color = '#888';
                fpsText.textContent = '0';
                stopWordUpdates();
            }
        } catch (error) {
            console.error('Stop camera error:', error);
            alert('Failed to stop camera.');
        }
    });

    // Status polling (FPS)
    setInterval(async () => {
        if (!cameraSection.classList.contains('hidden')) {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                fpsText.textContent = data.fps;
            } catch (error) {
                // Silently fail
            }
        }
    }, 1000);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !cameraSection.classList.contains('hidden')) {
            stopButton.click();
        }
        
        if (e.key === ' ' && !cameraSection.classList.contains('hidden')) {
            e.preventDefault();
            toggleButton.click();
        }
    });

    // Initialize
    showPage('home');
});