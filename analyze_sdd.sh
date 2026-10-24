#!/usr/bin/env bash
# analyze_sdd.sh
# Analyze SDD dataset structure and extract useful information for video encoder usage.
# Uses ffprobe to probe video files (requires ffmpeg tools).
# Outputs scene-wise video properties, suggested strides, and file counts.

set -euo pipefail

# Configuration
SDD_DIR="${HOME}/Desktop/Desktop/Datasets/SDD"
# Desired sampling rates (frames per second) for feature extraction
DESIRED_FPS=(1.0 0.5)  # You can adjust these
# Minimum frames to consider a video valid
MIN_FRAMES=10

# Check directory existence
if [[ ! -d "$SDD_DIR" ]]; then
    echo "Error: SDD directory not found at $SDD_DIR" >&2
    exit 1
fi

echo "=== SDD Dataset Analysis ==="
echo "Dataset root: $SDD_DIR"
echo ""

# Determine scenes from annotations directory (or videos)
if [[ -d "$SDD_DIR/annotations" ]]; then
    SCENES=($(ls -1 "$SDD_DIR/annotations"))
else
    # Fallback to videos directory
    SCENES=($(ls -1 "$SDD_DIR/videos"))
fi

if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "No scenes found in annotations or videos directories." >&2
    exit 1
fi

echo "Found ${#SCENES[@]} scenes: ${SCENES[*]}"
echo ""

# Overall stats
TOTAL_VIDEOS=0
TOTAL_DURATION_SEC=0

# Process each scene
for SCENE in "${SCENES[@]}"; do
    echo "--- Scene: $SCENE ---"

    # Paths
    ANNOT_SCENE_DIR="$SDD_DIR/annotations/$SCENE"
    VIDEO_SCENE_DIR="$SDD_DIR/videos/$SCENE"

    # Check existence
    if [[ ! -d "$ANNOT_SCENE_DIR" ]]; then
        echo "  WARNING: Annotations directory missing: $ANNOT_SCENE_DIR"
    fi
    if [[ ! -d "$VIDEO_SCENE_DIR" ]]; then
        echo "  WARNING: Videos directory missing: $VIDEO_SCENE_DIR"
        continue
    fi

    # List video subdirectories (video0, video1, ...)
    VIDEO_SUBS=($(ls -1d "$VIDEO_SCENE_DIR"/video* 2>/dev/null || true))
    if [[ ${#VIDEO_SUBS[@]} -eq 0 ]]; then
        echo "  No video subdirectories found in $VIDEO_SCENE_DIR"
        continue
    fi

    echo "  Found ${#VIDEO_SUBS[@]} video subdirectories"

    SCENE_VIDEO_COUNT=0
    SCENE_DURATION_SEC=0

    # Process each video subdirectory
    for VIDEO_SUB in "${VIDEO_SUBS[@]}"; do
        VIDEO_NAME=$(basename "$VIDEO_SUB")
        VIDEO_FILE="$VIDEO_SUB/video.mov"
        ANNOT_FILE="$ANNOT_SCENE_DIR/$VIDEO_NAME/annotations.txt"
        REF_FILE="$ANNOT_SCENE_DIR/$VIDEO_NAME/reference.jpg"

        # Check video file
        if [[ ! -f "$VIDEO_FILE" ]]; then
            echo "    $VIDEO_NAME: video.mov not found"
            continue
        fi

        # Probe video with ffmpeg
        echo "    $VIDEO_NAME: probing with ffmpeg..."
        # Default values (invalid)
        width=0
        height=0
        fps=0
        duration=0
        frames=0

        if FFMPEG_OUT=$(ffmpeg -hide_banner -loglevel error -i "$VIDEO_FILE" 2>&1); then
            # Extract duration: look for "Duration: HH:MM:SS.mmm"
            if echo "$FFMPEG_OUT" | grep -q "Duration:"; then
                duration_str=$(echo "$FFMPEG_OUT" | sed -n 's/.*Duration: \([^,]*\), start: .*/\1/p')
                # Convert HH:MM:SS.mmm to seconds
                IFS=: read -r h m s <<< "$duration_str"
                # s may contain decimal
                duration=$(echo "$h*3600 + $m*60 + $s" | bc)
            fi
            # Extract video stream: look for "Video: ... WxH, ... fps"
            if echo "$FFMPEG_OUT" | grep -q "Video:"; then
                video_line=$(echo "$FFMPEG_OUT" | sed -n '/Video:/p')
                # Extract resolution: WxH
                if echo "$video_line" | grep -E '[0-9]+x[0-9]+' >/dev/null; then
                    res=$(echo "$video_line" | grep -o '[0-9]\+x[0-9]\+' | head -1)
                    width=${res%x*}
                    height=${res#*x}
                fi
                # Extract fps: look for number before " fps"
                if echo "$video_line" | grep -E '[0-9]+\.?[0-9]* fps' >/dev/null; then
                    fps_str=$(echo "$video_line" | grep -o '[0-9]\+\.?[0-9]*\s*fps' | sed 's/ fps//')
                    fps=$(echo "$fps_str" | awk '{printf "%f", $1}')
                fi
            fi
        else
            echo "      ERROR: ffmpeg failed to probe the file"
            continue
        fi

        # Validate we got something reasonable
        if [[ "$width" -eq 0 || "$height" -eq 0 || $(echo "$duration == 0" | bc -l) -eq 1 ]]; then
            echo "      ERROR: Could not determine video properties (width/height/duration missing)."
            echo "      ffmpeg output (first 20 lines):"
            echo "$FFMPEG_OUT" | head -20
            continue
        fi

        echo "      Resolution: ${width}x${height}"
        echo "      FPS: $(printf "%.3f" "$fps")"
        echo "      Duration: ${duration}s"

        # Estimate frames if we have fps and duration
        if [ "$(echo "$fps > 0" | bc -l)" -eq 1 ] && [ "$(echo "$duration > 0" | bc -l)" -eq 1 ]; then
            frames=$(echo "$duration * $fps" | bc -l | cut -d'.' -f1)
        fi
        echo "      Estimated frames: $frames"

        # Check annotation and reference
        if [[ -f "$ANNOT_FILE" ]]; then
            ANNOT_LINES=$(wc -l < "$ANNOT_FILE")
            echo "      Annotations: $ANNOT_LINES lines"
        else
            echo "      Annotations: MISSING"
        fi
        if [[ -f "$REF_FILE" ]]; then
            echo "      Reference image: present"
        else
            echo "      Reference image: MISSING"
        fi

        # Suggest strides for desired FPS
        echo "      Suggested stride for feature extraction:"
        for DFPS in "${DESIRED_FPS[@]}"; do
            # stride = video_fps / desired_fps
            if [ "$(echo "$fps > 0" | bc -l)" -eq 1 ]; then
                STRIDE=$(echo "scale=0; $fps / $DFPS" | bc)
                # Ensure at least 1
                if [ "$STRIDE" -eq 0 ]; then
                    STRIDE=1
                fi
                echo "        - $DFPS fps -> stride ≈ $STRIDE (extract 1 frame every $STRIDE frames)"
            else
                echo "        - $DFPS fps -> cannot compute (invalid FPS)"
            fi
        done

        # Update counters
        ((SCENE_VIDEO_COUNT++))
        SCENE_DURATION_SEC=$(echo "$SCENE_DURATION_SEC + $duration" | bc)
    done

    echo "  Scene totals: $SCENE_VIDEO_COUNT videos, total duration $(echo "$SCENE_DURATION_SEC" | awk '{printf "%.1f", $1}')s"
    echo ""

    # Update overall counters
    TOTAL_VIDEOS=$((TOTAL_VIDEOS + SCENE_VIDEO_COUNT))
    TOTAL_DURATION_SEC=$(echo "$TOTAL_DURATION_SEC + $SCENE_DURATION_SEC" | bc)
done

echo "=== Overall Summary ==="
echo "Total scenes processed: ${#SCENES[@]}"
echo "Total videos found: $TOTAL_VIDEOS"
echo "Total duration: $(echo "$TOTAL_DURATION_SEC" | awk '{printf "%.1f", $1}')s ($(echo "$TOTAL_DURATION_SEC / 3600" | bc -l | awk '{printf "%.2f", $1}') hours)"
echo ""

# Additional helpful notes
echo "=== Notes for Video Encoder Usage ==="
echo "1. The video encoder expects frames extracted as JPEG files in <scene>/visual_data/ with 6-digit zero-padded names (frame000001.jpg, etc.)"
echo "2. Trajectory files should be placed in <data_root>/raw/all_data/<scene>.txt with format: frame_id ped_id x y (world coordinates)"
echo "3. You will need to convert SDD annotations (bounding boxes in image pixels) to world coordinates using the provided homography matrices for each video."
echo "4. Consider using a stride that gives you ~0.25-0.5 Hz sampling (similar to ETH/UCY preprocessing) unless you have specific temporal resolution needs."
echo "5. If you plan to use the agent-centric encoder, you'll need to extract crops around pedestrian positions; ensure world-to-pixel conversion matches your homography."
echo ""
echo "Analysis complete."