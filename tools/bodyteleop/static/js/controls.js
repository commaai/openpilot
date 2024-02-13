const keyVals = {w: 0, a: 0, s: 0, d: 0}

export function getXY() {
  let x = -keyVals.w + keyVals.s
  let y = -keyVals.d + keyVals.a
  return {x, y}
}

export const handleKeyX = (key, setValue) => {
  if (['w', 'a', 's', 'd'].includes(key)){
    keyVals[key] = setValue;
    let color = "#333";
    if (setValue === 1){
      color = "#e74c3c";
    }
    $("#key-"+key).css('background', color);
    const {x, y} = getXY();
    $("#pos-vals").text(x+","+y);
  }
};

export async function executePlan() {
  let plan = $("#plan-text").val();
  const planList = [];
  plan.split("\n").forEach(function(e){
    let line = e.split(",").map(k=>parseInt(k));
    if (line.length != 5 || line.slice(0, 4).map(e=>[1, 0].includes(e)).includes(false) || line[4] < 0 || line[4] > 10){
      console.log("invalid plan");
    }
    else{
      planList.push(line)
    }
  });

  async function execute() {
    for (var i = 0; i < planList.length; i++) {
      let [w, a, s, d, t] = planList[i];
      while(t > 0){
        console.log(w, a, s, d, t);
        if(w==1){$("#key-w").mousedown();}
        if(a==1){$("#key-a").mousedown();}
        if(s==1){$("#key-s").mousedown();}
        if(d==1){$("#key-d").mousedown();}
        await sleep(50);
        $("#key-w").mouseup();
        $("#key-a").mouseup();
        $("#key-s").mouseup();
        $("#key-d").mouseup();
        t = t - 0.05;
      }
    }
  }
  execute();
}