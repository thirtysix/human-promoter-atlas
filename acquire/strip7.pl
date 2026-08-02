#!/usr/bin/perl
# Strip a ChIP-Atlas assembled peak bed (Oth.ALL.05.<TF>.AllCell.bed) to 7 columns:
#   chrom  start  end  TF  score  SRX(experiment)  cell_group
# Drops the track header and the bulky annotation (Name/Title/antibody/source/strand/thick/rgb).
# Usage: <stream> | perl strip7.pl <TF>
use strict; use warnings;
my $tf = shift // "";
while (<STDIN>) {
    next if /^track/;
    chomp;
    my @F = split /\t/;
    next unless @F >= 5;
    my ($srx) = (($F[3] // "") =~ /(SRX\d+)/);
    my ($cg)  = (($F[3] // "") =~ /[Cc]ell%20group=([^;]+)/);
    print join("\t", $F[0], $F[1], $F[2], $tf, ($F[4] // ""), ($srx // ""), ($cg // "")), "\n";
}
