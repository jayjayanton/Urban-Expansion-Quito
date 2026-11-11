//VERSION=3

function setup() {
return {
input: [{
    bands: [
    "B02", "B03", "B04",
    "B08", "B11", "B12",
    "SCL", "dataMask"
    ],
    units: [
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "DN",
        "DN"
    ]
}],
mosaicking: "ORBIT",
output: {
    bands: 7,
    sampleType: "FLOAT32"
}
};
}

function evaluatePixel(samples) {
let values = [[], [], [], [], [], []];

for (let s of samples) {
// Valid scene classification classes:
// 4 = vegetation
// 5 = bare / non-vegetated
// 6 = water
// 7 = unclassified
if (s.dataMask !== 1 ||
    ![4, 5, 6, 7].includes(s.SCL)) {
    continue;
}

let bands = [
    s.B02, s.B03, s.B04,
    s.B08, s.B11, s.B12
];

for (let i = 0; i < 6; i++) {
    values[i].push(bands[i]);
}
}

function median(arr) {
if (arr.length === 0) return -9999;
arr.sort((a, b) => a - b);
let mid = Math.floor(arr.length / 2);

return arr.length % 2
    ? arr[mid]
    : (arr[mid - 1] + arr[mid]) / 2;
}

let count = values[0].length;

return [
median(values[0]),
median(values[1]),
median(values[2]),
median(values[3]),
median(values[4]),
median(values[5]),
count
];
}