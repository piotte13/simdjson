class ChunkWriter
    def initialize(base_filename="baseline-", file_size=640*1000, block_size=64)
        puts "#{file_size}"
        @@base_filename = base_filename
        @@file_size = file_size
        @@block_size = block_size
    end

    def prepare_chunk(chunks, include_newline)
        Array(chunks).map do |chunk|
            "#{chunk}#{' '*(@@block_size-chunk.bytesize-1)}#{include_newline ? "\n" : " "}"
        end.join("")
    end

    def write_files(filename, start1, repeat1, end1, repeat2: '', include_newline: true)
        start1  = prepare_chunk(start1, include_newline)
        repeat1 = prepare_chunk(repeat1, include_newline)
        end1    = prepare_chunk(end1, include_newline)

        write_full("#{@@base_filename}#{filename}.json", start1, repeat1, end1)

        if repeat2
            repeat2 = prepare_chunk(repeat2, include_newline)
            repeat2 = repeat2 * (repeat1.bytesize/repeat2.bytesize)
            write_half("#{@@base_filename}#{filename}-half.json", start1, repeat1, end1, repeat2)
            write_half_flip("#{@@base_filename}#{filename}-half-flip.json", start1, repeat1, end1, repeat2)
        end
    end

    def write_full(filename, start1, repeat1, end1)
        puts "Writing #{filename} ..."
        File.open(filename, "w") do |file|
            write_chunks(file, start1, repeat1, end1, @@file_size)
        end
        raise "OMG wrong file size #{File.size(filename)} (should be #{@@file_size})" if File.size(filename) != @@file_size
    end

    def write_half(filename, start1, repeat1, end1, repeat2)
        # repeat1 is already represented in start1 and end1, so it doesn't need quite
        # half the iterations.
        repeat1_len = (@@file_size/2) - start1.bytesize - end1.bytesize
        halfway_point = start1.bytesize + repeat1_len + repeat2.bytesize

        puts "Writing #{filename} ..."
        File.open(filename, "w") do |file|
            write_chunks(file, start1,  repeat1, repeat2, halfway_point)
            write_chunks(file, repeat2, repeat2, end1,    @@file_size-halfway_point)
        end
        raise "OMG wrong file size #{File.size(filename)} (should be #{@@file_size})" if File.size(filename) != @@file_size
    end

    def write_half_flip(filename, start1, repeat1, end1, repeat2)
        seed = case [start1.bytesize, repeat1.bytesize, end1.bytesize]
            # NOTE: assuming 1 start1 and 1 end block, the following seeds always yield exactly 5000 of
            # each block type and 2500 flips total (assuming the start1 and end blocks are the *second*
            # block type):
            # 5503, 26432, 31896, 45273, 67461, 67704, 77797, 104461, 108558, 161537, 174655, 177637,
            # 199709, 223227, 266237, 267190, 309059, 310992, 337453, 372029, 382165, 390469, 409164,
            # 410767, 415168, 421261, 427006, 431881, 449081, 452176, 464629, 471454, 471821, 485314,
            # 488718, 492301, 496234, 501446, 539483, 541109, 556822, 557334, 570479, 579819, 588407,
            # 596517, 600473, 607934, 626782, 631907, 641894, 657025, 683627, 699071, 706697, 713556,
            # 716590, 725729, 726031, 734363, 735574, 790114, 801675, 826308, 839923, 843554, 843770,
            # 847321, 856936, 879152, 883980, 905284, 926812, 955255, 960733, 971825, 987956
        when [64,64,64],[64,64,128]
            5503
        when [128,128,256]
            12851
        else
            raise "Unsupported block sizes [#{start1.bytesize},#{repeat1.bytesize},#{end1.bytesize}"
        end

        repeat_end = @@file_size-end1.bytesize
        random = Random.new(seed)
        percent_flips = 0.25*(repeat2.bytesize/64)

        puts "Writing #{filename} ..."
        File.open(filename, "w") do |file|
            pos = 0

            file.write(start1)
            pos += start1.bytesize

            current_repeat = repeat1
            loop do
                # Flip 1/4 of the time (but not at the beginning)
                if random.rand < percent_flips
                    current_repeat = (current_repeat == repeat1 ? repeat2 : repeat1)
                end
                break if pos+current_repeat.bytesize > repeat_end

                file.write(current_repeat)
                pos += current_repeat.bytesize
            end

            file.write(end1)
            pos += end1.bytesize
        end
        raise "OMG wrong file size #{File.size(filename)} (should be #{@@file_size})" if File.size(filename) != @@file_size
    end

    def write_chunks(file, start1, repeat1, end1, size)
        pos = 0
        file.write(start1)
        pos += start1.bytesize

        repeat_end = size-end1.bytesize
        loop do
            file.write(repeat1)
            pos += repeat1.bytesize
            break if pos >= repeat_end
        end

        file.write(end1)
        pos += end1.bytesize
        return pos
    end
end


w = ChunkWriter.new(*ARGV)
w.write_files "utf-8",          '["€"', ',"€"', ',"€"]'
w.write_files "0-structurals",  '0', '',  '', repeat2: nil
w.write_files "1-structurals",  [ '[', '0' ], [ ',', '0' ], [ ',', '{', '}', ']' ]
w.write_files "2-structurals",  '[0', ',0', [',{', '}]']
w.write_files "3-structurals",  '[{}', ',{}', ',0]'
w.write_files "4-structurals",  '[0,0', ',0,0', ',{}]'
w.write_files "5-structurals",  '[0,{}', ',0,{}', ',0,0]'
w.write_files "6-structurals",  '[0,0,0', ',0,0,0', ',0,{}]'
w.write_files "7-structurals",  '[0,0,{}', ',0,0,{}', ',0,0,0]'
w.write_files "8-structurals",  '[0,0,0,0', ',0,0,0,0', ',0,0,{}]'
w.write_files "9-structurals",  '[0,0,0,{}', ',0,0,0,{}', ',0,0,0,0]'
w.write_files "10-structurals", '[0,0,0,0,0', ',0,0,0,0,0', ',0,0,0,{}]'
w.write_files "11-structurals", '[0,0,0,0,{}', ',0,0,0,0,{}', ',0,0,0,0,0]'
w.write_files "12-structurals", '[0,0,0,0,0,0', ',0,0,0,0,0,0', ',0,0,0,0,{}]'
w.write_files "13-structurals", '[0,0,0,0,0,{}', ',0,0,0,0,0,{}', ',0,0,0,0,0,0]'
w.write_files "14-structurals", '[0,0,0,0,0,0,0', ',0,0,0,0,0,0,0', ',0,0,0,0,0,{}]'
w.write_files "15-structurals", '[0,0,0,0,0,0,{}', ',0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0]'
w.write_files "16-structurals", '[0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,{}]'
w.write_files "17-structurals", '[0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0]'
w.write_files "18-structurals", '[0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,{}]'
w.write_files "19-structurals", '[0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,0]'
w.write_files "20-structurals", '[0,0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0,{}]'
w.write_files "21-structurals", '[0,0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,0,0]'
w.write_files "22-structurals", '[0,0,0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0,0,0,0', ',0,0,0,0,0,0,0,0,0,{}]'
w.write_files "23-structurals", '[0,0,0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,0,0,{}', ',0,0,0,0,0,0,0,0,0,0,0]'
