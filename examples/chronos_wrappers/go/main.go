package main

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../build -lcutechronos_bridge -Wl,-rpath,${SRCDIR}/../build
#include "cutechronos_bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unsafe"
)

type result struct {
	Language          string    `json:"language"`
	Backend           string    `json:"backend"`
	ModelID           string    `json:"model_id"`
	Device            string    `json:"device"`
	PredictionLength  int       `json:"prediction_length"`
	Runs              int       `json:"runs"`
	Warmup            int       `json:"warmup"`
	AvgOuterLatencyMS float64   `json:"avg_outer_latency_ms"`
	AvgInnerLatencyMS float64   `json:"avg_inner_latency_ms"`
	MAE               float64   `json:"mae"`
	MAPEPct           float64   `json:"mape_pct"`
	Forecast          []float32 `json:"forecast"`
	Actual            []float32 `json:"actual"`
}

func parseCSVFloats(raw string) ([]float32, error) {
	parts := strings.Split(raw, ",")
	values := make([]float32, 0, len(parts))
	for _, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
		if err != nil {
			return nil, err
		}
		values = append(values, float32(value))
	}
	return values, nil
}

func computeMAE(pred []float32, actual []float32) float64 {
	length := len(pred)
	if len(actual) < length {
		length = len(actual)
	}
	if length == 0 {
		return 0
	}
	total := 0.0
	valid := 0
	for i := 0; i < length; i++ {
		if math.IsNaN(float64(pred[i])) || math.IsNaN(float64(actual[i])) {
			continue
		}
		total += math.Abs(float64(pred[i] - actual[i]))
		valid++
	}
	if valid == 0 {
		return 0
	}
	return total / float64(valid)
}

func computeMAPEPct(pred []float32, actual []float32) float64 {
	length := len(pred)
	if len(actual) < length {
		length = len(actual)
	}
	if length == 0 {
		return 0
	}
	total := 0.0
	valid := 0
	for i := 0; i < length; i++ {
		if math.IsNaN(float64(pred[i])) || math.IsNaN(float64(actual[i])) || math.Abs(float64(actual[i])) < 1e-12 {
			continue
		}
		total += math.Abs(float64((pred[i]-actual[i])/actual[i])) * 100.0
		valid++
	}
	if valid == 0 {
		return 0
	}
	return total / float64(valid)
}

func main() {
	runtime.LockOSThread()

	modelID := flag.String("model-id", "amazon/chronos-2", "")
	backend := flag.String("backend", "cute", "")
	device := flag.String("device", "cuda", "")
	dtype := flag.String("dtype", "bfloat16", "")
	compileMode := flag.String("compile-mode", "", "")
	contextCSV := flag.String("context", "", "")
	actualCSV := flag.String("actual", "", "")
	predictionLength := flag.Int("prediction-length", 3, "")
	runs := flag.Int("runs", 5, "")
	warmup := flag.Int("warmup", 1, "")
	flag.Parse()

	if *contextCSV == "" || *actualCSV == "" {
		fmt.Fprintln(os.Stderr, "both --context and --actual are required")
		os.Exit(2)
	}

	context, err := parseCSVFloats(*contextCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "context parse failed: %v\n", err)
		os.Exit(1)
	}
	actual, err := parseCSVFloats(*actualCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "actual parse failed: %v\n", err)
		os.Exit(1)
	}

	var handle C.int
	errorBuffer := make([]byte, 4096)
	cModelID := C.CString(*modelID)
	cBackend := C.CString(*backend)
	cDevice := C.CString(*device)
	cDType := C.CString(*dtype)
	var cCompileMode *C.char
	if *compileMode != "" {
		cCompileMode = C.CString(*compileMode)
	}
	defer C.free(unsafe.Pointer(cModelID))
	defer C.free(unsafe.Pointer(cBackend))
	defer C.free(unsafe.Pointer(cDevice))
	defer C.free(unsafe.Pointer(cDType))
	if cCompileMode != nil {
		defer C.free(unsafe.Pointer(cCompileMode))
	}

	rc := C.cutechronos_init_pipeline(
		cModelID,
		cBackend,
		cDevice,
		cDType,
		cCompileMode,
		&handle,
		(*C.char)(unsafe.Pointer(&errorBuffer[0])),
		C.size_t(len(errorBuffer)),
	)
	if rc != 0 {
		fmt.Fprintf(os.Stderr, "init failed: %s\n", strings.TrimRight(string(errorBuffer), "\x00"))
		os.Exit(1)
	}
	defer C.cutechronos_destroy_pipeline(
		handle,
		(*C.char)(unsafe.Pointer(&errorBuffer[0])),
		C.size_t(len(errorBuffer)),
	)

	forecastBuffer := make([]float32, *predictionLength)
	var forecastLen C.int
	var innerMS C.double

	for i := 0; i < *warmup; i++ {
		rc = C.cutechronos_predict_median(
			handle,
			(*C.float)(unsafe.Pointer(&context[0])),
			C.int(len(context)),
			C.int(*predictionLength),
			(*C.float)(unsafe.Pointer(&forecastBuffer[0])),
			C.int(len(forecastBuffer)),
			&forecastLen,
			&innerMS,
			(*C.char)(unsafe.Pointer(&errorBuffer[0])),
			C.size_t(len(errorBuffer)),
		)
		if rc != 0 {
			fmt.Fprintf(os.Stderr, "warmup failed: %s\n", strings.TrimRight(string(errorBuffer), "\x00"))
			os.Exit(1)
		}
	}

	totalOuterMS := 0.0
	totalInnerMS := 0.0
	for i := 0; i < *runs; i++ {
		start := time.Now()
		rc = C.cutechronos_predict_median(
			handle,
			(*C.float)(unsafe.Pointer(&context[0])),
			C.int(len(context)),
			C.int(*predictionLength),
			(*C.float)(unsafe.Pointer(&forecastBuffer[0])),
			C.int(len(forecastBuffer)),
			&forecastLen,
			&innerMS,
			(*C.char)(unsafe.Pointer(&errorBuffer[0])),
			C.size_t(len(errorBuffer)),
		)
		if rc != 0 {
			fmt.Fprintf(os.Stderr, "predict failed: %s\n", strings.TrimRight(string(errorBuffer), "\x00"))
			os.Exit(1)
		}
		totalOuterMS += float64(time.Since(start).Microseconds()) / 1000.0
		totalInnerMS += float64(innerMS)
	}

	forecast := append([]float32(nil), forecastBuffer[:int(forecastLen)]...)
	output := result{
		Language:          "go",
		Backend:           *backend,
		ModelID:           *modelID,
		Device:            *device,
		PredictionLength:  *predictionLength,
		Runs:              *runs,
		Warmup:            *warmup,
		AvgOuterLatencyMS: totalOuterMS / float64(*runs),
		AvgInnerLatencyMS: totalInnerMS / float64(*runs),
		MAE:               computeMAE(forecast, actual),
		MAPEPct:           computeMAPEPct(forecast, actual),
		Forecast:          forecast,
		Actual:            actual,
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "json encode failed: %v\n", err)
		os.Exit(1)
	}
}
