// ---------------- STATE ----------------
let predData = [], actualData = [], labels = [];
let maxPoints = 20;
let driftCount = 0;
let prevDriftCount = 0;
// ---------------- WINDOW CONTROL ----------------
function setWindow(n){
    maxPoints = n;
}


// ---------------- CHART INIT ----------------
const chart = new Chart(document.getElementById('chart'), {
    type: 'line',
    data: {
        labels,
        datasets: [
            {
                label:'Predicted Demand',
                data:predData,
                borderColor:'#22d3ee',
                tension:0.4
            },
            {
                label:'Actual Demand',
                data:actualData,
                borderColor:'#4ade80',
                tension:0.4
            }
        ]
    },
    options:{
        animation:{duration:400},
        plugins:{legend:{labels:{color:'#cbd5f5'}}},
        scales:{
            x:{ticks:{color:'#94a3b8'}},
            y:{ticks:{color:'#94a3b8'}}
        }
    }
});

// ---------------- SPARKLINES ----------------
const sparkPred = new Chart(document.getElementById('sparkPred'), {
    type:'line',
    data:{labels, datasets:[{data:predData, borderColor:'#22d3ee'}]},
    options:{
        plugins:{legend:{display:false}},
        scales:{x:{display:false},y:{display:false}}
    }
});

const sparkActual = new Chart(document.getElementById('sparkActual'), {
    type:'line',
    data:{labels, datasets:[{data:actualData, borderColor:'#4ade80'}]},
    options:{
        plugins:{legend:{display:false}},
        scales:{x:{display:false},y:{display:false}}
    }
});

// ---------------- UI EFFECT ----------------
function pulse(id,val){
    const el=document.getElementById(id);
    el.style.transform='scale(1.1)';
    setTimeout(()=>el.style.transform='scale(1)',150);
    el.innerText=val;
}

// ---------------- DRIFT HANDLER ----------------
function handleDrift(newDriftCount){

    const driftEl = document.getElementById('drift');
    const statusEl = document.getElementById('status');
    const alertEl = document.getElementById('alert');
    const counterEl = document.getElementById('driftCount');

    // Always show current counter
    counterEl.innerText = newDriftCount + " / 3";

    // 🔥 ONLY trigger when drift_count increases
    if(newDriftCount > prevDriftCount){

        // Flash drift warning
        driftEl.innerText = 'HIGH ⚠️';
        driftEl.className = 'text-4xl font-bold mt-2 text-red-500';

        statusEl.innerText = 'Drift Detected';
        statusEl.className = 'text-xl mt-2 text-yellow-400';

        alertEl.innerText = '⚠️ Concept Drift Detected';

        // 🔁 revert back after short time (instant event feel)
        setTimeout(() => {
            driftEl.innerText = 'STABLE';
            driftEl.className = 'text-4xl font-bold mt-2 text-green-400';

          //  statusEl.innerText = 'Model Healthy';
           // statusEl.className = 'text-xl mt-2 text-green-400';

            alertEl.innerText = '';
        }, 700);
    }

    // 🔥 Retraining trigger (purely based on backend count)
    if(newDriftCount >= 3){
        statusEl.innerText = 'Retraining Model...';
        statusEl.className = 'text-xl mt-2 text-red-400';
          
    }

    if(newDriftCount === 0) {
        statusEl.innerText = 'Model Healthy';
        statusEl.className = 'text-xl mt-2 text-green-400';
    }

    // Update previous count
    prevDriftCount = newDriftCount;
}

// ---------------- FETCH LOOP ----------------
async function fetchData(){

    try{
        const res = await fetch('../outputs/live.json?t=' + Date.now());

        if(!res.ok){
            throw new Error("Failed to fetch live.json");
        }

        const data = await res.json();

        // 🔢 Update UI numbers
        pulse('pred', data.predicted.toFixed(2));
        pulse('actual', data.actual.toFixed(2));

        // 🔥 Drift handled ONLY via drift_count
        handleDrift(data.drift_count);

        // 📊 Update chart data
        const timeLabel = new Date().toLocaleTimeString();

        labels.push(timeLabel);
        predData.push(data.predicted);
        actualData.push(data.actual);

        // Sliding window
        if(labels.length > maxPoints){
            labels.shift();
            predData.shift();
            actualData.shift();
        }

        // Update charts
        chart.update();
        sparkPred.update();
        sparkActual.update();

    } catch(e){
        console.error("Fetch Error:", e);
    }
}
// ---------------- START ----------------
setInterval(fetchData, 1000);