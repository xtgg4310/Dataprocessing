import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg

if __name__ == '__main__':
    data_time = txt.txt_to_matrix(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\signal_updown.txt")
    data, time, rate = RPI.pre_process_RPI(data_time)
    timestamp = RPI.pre_process_time(time)
    data_new = sg.butter_highpass_filter(data, 1, rate, 5)
    fft_data, s_rate, r_rate = sg.fft_data(data_new, timestamp, 0, len(data))
    sg.time_phase_draw(data_new, rate, None, timestamp)
    sg.signal_draw_save(data, timestamp, fft_data, s_rate, 0, len(data))
    sg.stft_scalar_calculate(data_new, rate,50)
