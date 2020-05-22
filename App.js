import React, { useEffect, useState, useRef } from "react";
import {
  TextInput,
  StyleSheet,
  Text,
  View,
  Button,
  Image,
  Dimensions,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import { ImageBrowser } from "expo-multiple-media-imagepicker";
import * as tf from "@tensorflow/tfjs";
import * as tf_rn from "@tensorflow/tfjs-react-native";
import * as Permissions from "expo-permissions";
import * as jpeg from "jpeg-js";
import { Col, Row, Grid } from "react-native-easy-grid";

const askPerm = async () => {
  return await Permissions.askAsync(Permissions.CAMERA_ROLL);
};

function chunk(myArray, chunk_size) {
  var index = 0;
  var arrayLength = myArray.length;
  var tempArray = [];
  let myChunk;
  for (index = 0; index < arrayLength; index += chunk_size) {
    myChunk = myArray.slice(index, index + chunk_size);
    tempArray.push(myChunk);
  }

  return tempArray;
}
function imageToTensor(rawImageData) {
  const TO_UINT8ARRAY = true;
  const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
  const buffer = new Uint8Array(width * height * 3);
  let offset = 0; // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset];
    buffer[i + 1] = data[offset + 1];
    buffer[i + 2] = data[offset + 2];

    offset += 4;
  }

  return tf.tensor3d(buffer, [height, width, 3]);
}

export default function App() {
  const [imagePickerOpen, setImagePickerOpen] = useState(false);
  const [TFReady, setTFReady] = useState(false);
  const [currentClass, setCurrentClass] = useState(null);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const imageTensorBatch = useRef([]);
  const labelTensorBatch = useRef([]);
  const [number, setNumber] = useState(0);
  const [step, setStep] = useState("data");
  const [model, setModel] = useState();

  useEffect(() => {
    askPerm();
    tf.ready().then(() => setTFReady(true));
    tf.setBackend("cpu");
    console.log(tf.getBackend());
  }, []);

  const imageBrowserCallback = async (s) => {
    let files = await s;
    files = files.filter((f) => !f.uri.endsWith("PNG"));
    setNumber(number + files.length);
    if (files.length < 2) {
      alert("Not Enough Files");
    } else {
      setData([...data, { name: currentClass, files: files }]);
      setCurrentClass(null);
      setImagePickerOpen(false);
    }
  };

  const fetchImage = async (uri) => {
    const r = await tf_rn.fetch(uri, {}, { isBinary: true });
    const rawImageData = await r.arrayBuffer();
    const imageTensor = imageToTensor(rawImageData);
    return imageTensor;
  };

  const loadData = async () => {
    setLoading(true);
    const numClasses = data.length;
    data.forEach((c, id) => {
      c.files.forEach(async (f) => {
        let onh = Array(numClasses).fill(0);
        onh[id] = 1;
        labelTensorBatch.current.push(tf.tensor1d(onh));
        imageTensorBatch.current.push(
          tf.image.resizeBilinear(await fetchImage(f.uri), [224, 224])
        );
        if (imageTensorBatch.current.length === number) {
          setStep("model");
          alert("Data Loaded");
          setLoading(false);
        }
      });
    });
  };

  const stackImages = () => {
    imageTensorBatch.current = tf.stack(imageTensorBatch.current);
    labelTensorBatch.current = tf.stack(labelTensorBatch.current);
    alert(";cool");
  };

  const trainModel = async () => {
    const h = await model.fit(
      imageTensorBatch.current,
      labelTensorBatch.current,
      {
        batchSize: 1,
        epochs: 3,
      }
    );
    console.log("Loss after Epoch  : " + h.history.loss[0]);
    setModel(model);
    alert("Sucess");
  };

  const compileModel = async () => {
    let m = tf.sequential();
    m.add(
      tf.layers.dense({
        inputShape: [224, 224, 3],
        units: 16,
        activation: "relu",
      })
    );
    m.add(tf.layers.flatten());
    m.add(
      tf.layers.dense({
        units: data.length,
        activation: "softmax",
      })
    );
    m.compile({
      optimizer: "adam",
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    });
    m.summary();
    setModel(m);
    alert("model compiled");
  };

  if (imagePickerOpen) {
    return (
      <ImageBrowser
        max={20} // Maximum number of pickable image. default is None
        headerButtonColor={"#E31676"} // Button color on header.
        badgeColor={"#E31676"} // Badge color when picking.
        callback={imageBrowserCallback}
      />
    );
  }
  if (loading) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text>Loading Images as Tensors</Text>
      </View>
    );
  }
  return (
    <View style={styles.container}>
      {step === "data" ? (
        <View>
          <Text>TF Loaded {TFReady.toString().toUpperCase()}</Text>
          <View>
            <View
              style={{
                backgroundColor: "#5844ed",
                borderRadius: 10,
                margin: 10,
                alignItems: "center",
              }}
            >
              <Text style={styles.formText}>Enter Name of class</Text>
              <TextInput
                style={styles.formInput}
                placeholder="Cat,Dog,.."
                onChangeText={(text) => setCurrentClass(text)}
              />
              <TouchableOpacity
                style={{
                  backgroundColor: "white",
                  padding: 6,
                  margin: 6,
                  borderRadius: "8",
                  width: "80%",
                  alignItems: "center",
                }}
                onPress={() =>
                  currentClass
                    ? setImagePickerOpen(true)
                    : alert("Enter Class name")
                }
              >
                <Text>Select Image</Text>
              </TouchableOpacity>
            </View>
          </View>
          <ScrollView>
            {data.map((d) => (
              <View key={d.name} style={styles.classPreview}>
                <Text
                  style={{ fontSize: 15, paddingBottom: 10, color: "white" }}
                >
                  Class Name : {d.name}
                </Text>
                <Text
                  style={{ fontSize: 15, paddingBottom: 10, color: "white" }}
                >
                  Files Selected: {d.files.length}
                </Text>
                {chunk(d.files, 4).map((row, i) => {
                  return (
                    <View key={i}>
                      <View style={{ flex: 1, flexDirection: "row" }}>
                        {row.map((b) => (
                          <Image
                            key={b.uri}
                            source={{ uri: b.uri }}
                            style={styles.imgPreview}
                          ></Image>
                        ))}
                      </View>
                    </View>
                  );
                })}
              </View>
            ))}
          </ScrollView>
          <Button
            disabled={data.length < 2}
            title="Load Data"
            onPress={loadData}
          ></Button>
        </View>
      ) : null}
      {step == "model" ? (
        <View>
          <Text>TODO Train model</Text>
          <Text>Total Images {imageTensorBatch.current.length}</Text>
          <Button onPress={compileModel} title="Compile model"></Button>
          <Button onPress={stackImages} title="stack"></Button>
          <Button onPress={trainModel} title="Train model"></Button>
          <Button
            onPress={() => {
              setStep("data");
              imageTensorBatch.current = [];
              setData([]);
            }}
            title="Reset"
          ></Button>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingTop: Dimensions.get("window").height * 0.05,
    paddingBottom: Dimensions.get("window").height * 0.05,
    flex: 1,
  },
  formText: {
    padding: 10,
    fontSize: 20,
    color: "white",
    fontWeight: "bold",
  },
  formInput: {
    backgroundColor: "white",
    padding: 10,
    height: 40,
    width: "80%",
    borderRadius: 10,
  },
  imgPreview: {
    width: (Dimensions.get("window").width - 20) * 0.25,
    height: (Dimensions.get("window").width - 20) * 0.25,
  },
  classPreview: {
    backgroundColor: "#2f9c0a",
    margin: 6,
    padding: 4,
    borderRadius: 10,
  },
});
