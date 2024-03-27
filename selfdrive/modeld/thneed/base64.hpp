#ifndef BASE64_HPP_
#define BASE64_HPP_

#include <vector>
#include <string>
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(uint8_t c)
{
  return (isalnum(c) || (c == '+') || (c == '/'));
}

inline std::vector<uint8_t> base64_decode(std::string const &encoded_string)
{
  size_t input_length = encoded_string.size();
  int char_index = 0;
  int padding_index = 0;
  int index_of_current_character = 0;
  uint8_t char_array_4[4], char_array_3[3];
  std::vector<uint8_t> output;

  while (input_length-- && (encoded_string[index_of_current_character] != '=') && is_base64(encoded_string[index_of_current_character]))
  {
    char_array_4[char_index++] = encoded_string[index_of_current_character];
    index_of_current_character++;
    if (char_index == 4)
    {
      for (char_index = 0; char_index < 4; char_index++)
      {
        char_array_4[char_index] = base64_chars.find(char_array_4[char_index]);
      }

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (char_index = 0; (char_index < 3); char_index++)
      {
        output.push_back(char_array_3[char_index]);
      }
      char_index = 0;
    }
  }

  if (char_index)
  {
    for (padding_index = char_index; padding_index < 4; padding_index++)
    {
      char_array_4[padding_index] = 0;
    }
    for (padding_index = 0; padding_index < 4; padding_index++)
    {
      char_array_4[padding_index] = base64_chars.find(char_array_4[padding_index]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (padding_index = 0; (padding_index < char_index - 1); padding_index++)
    {
      output.push_back(char_array_3[padding_index]);
    }
  }

  return output;
}

#endif