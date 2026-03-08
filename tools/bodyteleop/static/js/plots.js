export const pingPoints = [];
export const batteryPoints = [];

function getChartConfig(pts, color, title, ymax=100) {
  return {
    type: 'line',
    data: {
      datasets: [{
        label: title,
        data: pts,
        borderWidth: 1,
        borderColor: color,
        backgroundColor: color,
        fill: 'origin'
      }]
    },
    options: {
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'minute',
            displayFormats: {
              second: 'h:mm a'
            }
          },
          grid: {
            color: '#222', // Grid lines color
          },
          ticks: {
            source: 'data',
            fontColor: 'rgba(255, 255, 255, 1.0)', // Y-axis label color
          }
        },
        y: {
          beginAtZero: true,
          max: ymax,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)', // Grid lines color
          },
          ticks: {
            fontColor: 'rgba(255, 255, 255, 0.7)', // Y-axis label color
          }
        }
      }
    }
  }
}

const ctxPing = document.getElementById('chart-ping');
const ctxBattery = document.getElementById('chart-battery');
export const chartPing = new Chart(ctxPing, getChartConfig(pingPoints, 'rgba(192, 57, 43, 0.7)', 'Controls Ping Time (ms)', 250));
export const chartBattery = new Chart(ctxBattery, getChartConfig(batteryPoints, 'rgba(41, 128, 185, 0.7)', 'Battery %', 100));
