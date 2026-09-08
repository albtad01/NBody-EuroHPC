if (NOT DEFINED MURB_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "MURB_EXECUTABLE and TEST_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(trajectory "${TEST_DIRECTORY}/recorded.murbtraj")
set(legacy_file "${TEST_DIRECTORY}/simulation_data.bin")
file(REMOVE "${trajectory}" "${legacy_file}")

execute_process(
    COMMAND "${MURB_EXECUTABLE}" -n 8 -i 2 --im cpu+naive --nv
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE no_record_result
    OUTPUT_VARIABLE no_record_output
    ERROR_VARIABLE no_record_error)
if (NOT no_record_result EQUAL 0)
    message(FATAL_ERROR "non-recording run failed: ${no_record_error}")
endif()
file(GLOB unexpected_trajectories "${TEST_DIRECTORY}/*.murbtraj")
if (unexpected_trajectories OR EXISTS "${legacy_file}")
    message(FATAL_ERROR "non-recording run created a trajectory file")
endif()

execute_process(
    COMMAND "${MURB_EXECUTABLE}" -n 8 -i 3 --warmup 1 --im cpu+naive --nv
            --record "${trajectory}" --record-every 2
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE record_result
    OUTPUT_VARIABLE record_output
    ERROR_VARIABLE record_error)
if (NOT record_result EQUAL 0 OR NOT EXISTS "${trajectory}")
    message(FATAL_ERROR "recording run failed: ${record_error}")
endif()
if (NOT record_output MATCHES "recorded_frames=1")
    message(FATAL_ERROR "recording run reported an unexpected frame count: ${record_output}")
endif()

execute_process(
    COMMAND "${MURB_EXECUTABLE}" --replay "${trajectory}" --nv
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE replay_result
    OUTPUT_VARIABLE replay_output
    ERROR_VARIABLE replay_error)
if (NOT replay_result EQUAL 0)
    message(FATAL_ERROR "headless replay failed: ${replay_error}")
endif()
if (NOT replay_output MATCHES "replayed_frames=1")
    message(FATAL_ERROR "replay reported an unexpected frame count: ${replay_output}")
endif()

file(REMOVE "${trajectory}")
