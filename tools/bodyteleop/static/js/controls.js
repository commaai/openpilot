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
